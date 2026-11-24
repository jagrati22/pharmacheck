# PharmaCheck — Drug Recall & Batch Safety RAG Watchdog

An AI-powered medicine batch safety verification platform built with **Retrieval-Augmented Generation (RAG)**, **MongoDB Atlas Vector Search**, **Gemini 2.5 Flash**, **FastAPI**, and **Next.js**.

## Key Features
- **Batch Checker & AI Strip Scanner:** Verify medicine names and batch numbers against live CDSCO (India) and US FDA records, or upload a medicine strip photo for automated OCR extraction via Gemini Vision.
- **Schema-Aware Record RAG:** Serializes each regulatory NSQ/recall entry into a self-contained semantic record embedded with `all-MiniLM-L6-v2` and indexed in MongoDB Atlas Vector Search.
- **Hybrid Retrieval + Grounded Explanations:** Combines exact regex batch matching with vector similarity search and generates patient-friendly explanations with official citations.
- **Active Alerts Index:** Searchable table of flagged drug batches, manufacturers, and lab test failures.
- **Chemists & RAG Sync Portal:** One-click live sync with OpenFDA API & CDSCO alerts plus custom state drug controller PDF ingestion.

## Tech Stack
- **Frontend:** Next.js (App Router), TypeScript, Tailwind CSS, Lucide Icons
- **Backend:** Python, FastAPI, Uvicorn, PyMuPDF (`fitz`)
- **AI & Embeddings:** Google Gemini 2.5 Flash API, Hugging Face `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Database:** MongoDB Atlas (Document Store + Vector Search)