import os
import requests
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from playwright.sync_api import sync_playwright

load_dotenv()

# 1. Connect to MongoDB Atlas (separate clean DB: pharmacheck_db)
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["pharmacheck_db"]
collection = db["drug_alerts"]

# 2. Load free local embedding model (384 dimensions)
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def fetch_openfda(limit=15):
    print("Fetching live recalls from OpenFDA...")
    url = f"https://api.fda.gov/drug/enforcement.json?limit={limit}"
    res = requests.get(url, timeout=20).json()
    records = []

    for item in res.get("results", []):
        drug_name = item.get("product_description", "Unknown Drug")[:150]
        batch_no = item.get("code_info", "N/A")
        manufacturer = item.get("recalling_firm", "Unknown Manufacturer")
        reason = item.get("reason_for_recall", "Regulatory Recall")
        date = item.get("recall_initiation_date", "N/A")

        records.append({
            "source": "OpenFDA (USA)",
            "drug_name": drug_name,
            "batch_no": batch_no,
            "manufacturer": manufacturer,
            "reason": reason,
            "date": date,
            "status": "RECALLED",
        })
    return records


def fetch_cdsco():
    print("Fetching live NSQ records from CDSCO India...")
    url = "https://cdscoonline.gov.in/CDSCO/viewPublicNSQDrug"
    records = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_context(ignore_https_errors=True).new_page()
        page.goto(url, timeout=45000)
        page.wait_for_timeout(5000)

        rows = page.query_selector_all("table tr")
        for r in rows[1:]:  # Skip header row
            cols = [c.inner_text().strip().replace("\n", " ") for c in r.query_selector_all("td")]
            if len(cols) >= 10:
                records.append({
                    "source": "CDSCO (India)",
                    "drug_name": cols[1],
                    "batch_no": cols[2],
                    "mfg_date": cols[3],
                    "exp_date": cols[4],
                    "manufacturer": cols[5],
                    "reason": cols[6],
                    "lab": cols[8],
                    "date": cols[9],
                    "status": "NSQ (Not of Standard Quality)",
                })
        browser.close()
    return records


def sync_database():
    fda_data = fetch_openfda(limit=15)
    cdsco_data = fetch_cdsco()
    all_records = cdsco_data + fda_data

    print(f"Total records fetched: {len(all_records)} (CDSCO: {len(cdsco_data)}, OpenFDA: {len(fda_data)})")

    # Clear old entries so re-running stays clean without duplicates
    collection.delete_many({})

    for rec in all_records:
        search_text = f"Drug: {rec['drug_name']} | Batch: {rec['batch_no']} | Manufacturer: {rec['manufacturer']} | Reason: {rec['reason']} | Source: {rec['source']}"
        rec["search_text"] = search_text
        rec["embedding"] = embedder.encode(search_text).tolist()

    if all_records:
        collection.insert_many(all_records)
        print("SUCCESS! All live regulatory records + vector embeddings saved in MongoDB Atlas!")


if __name__ == "__main__":
    sync_database()