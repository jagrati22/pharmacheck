import os
import re
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from google import genai
from ingest import sync_database

load_dotenv()

app = FastAPI(title="PharmaCheck RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = MongoClient(os.getenv("MONGODB_URI"))
db = client["pharmacheck_db"]
collection = db["drug_alerts"]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class SearchRequest(BaseModel):
    query: str


@app.get("/")
def home():
    total = collection.count_documents({})
    return {"status": "PharmaCheck RAG Backend Live", "total_alerts_indexed": total}


@app.get("/alerts")
def get_recent_alerts():
    docs = list(collection.find({}, {"_id": 0, "embedding": 0}).limit(25))
    return {"alerts": docs}


@app.post("/sync")
def trigger_live_sync():
    sync_database()
    total = collection.count_documents({})
    return {"status": "SYNCED", "total_alerts_indexed": total}


@app.post("/scan-strip")
async def scan_medicine_strip(file: UploadFile = File(...)):
    image_bytes = await file.read()
    mime_type = file.content_type or "image/jpeg"

    prompt = """Extract the Medicine/Brand Name and the Batch Number (Batch No / B.No / Lot No) from this medicine strip image.
Return ONLY valid JSON in this exact format without markdown backticks:
{"medicine_name": "...", "batch_number": "..."}
If either is not clearly visible, return an empty string "" for that field."""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ],
    )

    raw_text = response.text.strip().replace("```json", "").replace("```", "").strip()
    try:
        extracted = json.loads(raw_text)
    except Exception:
        extracted = {"medicine_name": "", "batch_number": ""}

    return extracted


@app.post("/upload-pdf")
async def upload_regulatory_pdf(file: UploadFile = File(...)):
    pdf_bytes = await file.read()

    prompt = """Extract all recalled or Not-of-Standard-Quality (NSQ) drug records from this regulatory PDF.
Return ONLY a valid JSON array of objects without markdown backticks, where each object has:
{
  "drug_name": "...",
  "batch_no": "...",
  "manufacturer": "...",
  "reason": "...",
  "date": "..."
}
If none are found, return []."""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            genai.types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            prompt,
        ],
    )

    raw_text = response.text.strip().replace("```json", "").replace("```", "").strip()
    try:
        records = json.loads(raw_text)
    except Exception:
        records = []

    inserted = 0
    for rec in records:
        drug = rec.get("drug_name", "").strip()
        batch = rec.get("batch_no", "").strip()
        if not drug:
            continue
        reason = rec.get("reason", "Regulatory Circular Alert")
        manufacturer = rec.get("manufacturer", "Unknown Manufacturer")
        search_text = f"Drug: {drug} | Batch: {batch} | Manufacturer: {manufacturer} | Reason: {reason} | Source: Uploaded PDF ({file.filename})"
        embedding = embedder.encode(search_text).tolist()

        collection.update_one(
            {"drug_name": drug, "batch_no": batch},
            {
                "$set": {
                    "source": f"PDF: {file.filename}",
                    "drug_name": drug,
                    "batch_no": batch or "N/A",
                    "manufacturer": manufacturer,
                    "reason": reason,
                    "date": rec.get("date", "Uploaded"),
                    "status": "NSQ / Recall Alert",
                    "search_text": search_text,
                    "embedding": embedding,
                }
            },
            upsert=True,
        )
        inserted += 1

    return {"status": "PDF_INDEXED", "records_added": inserted}


@app.post("/verify")
def verify_drug(req: SearchRequest):
    user_query = req.query.strip()
    if not user_query:
        return {"status": "EMPTY", "ai_verdict": "Please enter a medicine name or batch number.", "matches": []}

    # A. Exact / Regex Match
    safe_pattern = re.escape(user_query)
    exact_matches = list(
        collection.find(
            {
                "$or": [
                    {"batch_no": {"$regex": safe_pattern, "$options": "i"}},
                    {"drug_name": {"$regex": safe_pattern, "$options": "i"}},
                ]
            },
            {"_id": 0, "embedding": 0},
        ).limit(5)
    )

    # B. Semantic Vector Search (only keep relevant matches with score >= 0.62)
    query_vector = embedder.encode(user_query).tolist()
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 50,
                "limit": 5,
            }
        },
        {
            "$project": {
                "_id": 0,
                "embedding": 0,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    raw_vector_matches = list(collection.aggregate(pipeline))
    vector_matches = [m for m in raw_vector_matches if m.get("score", 0) >= 0.62]

    # Merge & deduplicate
    combined = []
    seen = set()
    for item in exact_matches + vector_matches:
        key = (item.get("drug_name"), item.get("batch_no"))
        if key not in seen:
            seen.add(key)
            combined.append(item)

    top_matches = combined[:5]

    # C. If no matching alert is found in database
    if not top_matches:
        return {
            "query": user_query,
            "status": "CLEAR",
            "ai_verdict": (
                f"No active recall or Not-of-Standard-Quality (NSQ) alert was found for '{user_query}' "
                "in our indexed CDSCO (India) and OpenFDA (USA) regulatory records. "
                "Always verify the physical expiry date, packaging seal, and consult a pharmacist or doctor before use."
            ),
            "matches": [],
        }

    # D. Generate Grounded Explanation with Gemini when alerts exist
    context_lines = []
    for i, m in enumerate(top_matches, 1):
        context_lines.append(
            f"[{i}] Source: {m.get('source')} | Drug: {m.get('drug_name')} | "
            f"Batch: {m.get('batch_no')} | Manufacturer: {m.get('manufacturer')} | "
            f"Status: {m.get('status')} | Reason: {m.get('reason')} | Date: {m.get('date')}"
        )
    context_str = "\n".join(context_lines)

    prompt = f"""You are PharmaCheck AI, a drug safety verification assistant.
User searched for: "{user_query}"

Retrieved Regulatory Records (CDSCO India & OpenFDA USA):
{context_str}

Instructions:
1. Check if the user's searched drug or batch matches any recalled/NSQ alert above.
2. Explain the reason for failure/recall in simple, patient-friendly language and cite the exact source ([1], [2], etc.).
3. If the retrieved records are only loosely related and NOT the user's searched drug, clearly state that no exact match was found.
4. Keep the response concise, accurate, and grounded ONLY in the records above."""

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return {
        "query": user_query,
        "status": "ALERT",
        "ai_verdict": response.text,
        "matches": top_matches,
    }