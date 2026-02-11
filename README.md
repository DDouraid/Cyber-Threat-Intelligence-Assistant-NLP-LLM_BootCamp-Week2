# Cyber Threat Intelligence Assistant (BootCamp Week 2)

A **voice-enabled cybersecurity threat intelligence assistant** that uses RAG (Retrieval-Augmented Generation) over CVE/NVD and MITRE ATT&CK data. Ask questions in natural language, get answers grounded in threat intelligence, and use optional features like image analysis, incident reports, log analysis, and file scanning.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green) ![React](https://img.shields.io/badge/React-19-61dafb)

---

## Features

- **RAG Q&A** – Ask cybersecurity questions; answers are based on retrieved CVE and MITRE ATT&CK context.
- **Guided mode** – Step-by-step explanations for non-technical users.
- **Vision RAG** – Upload an image; the assistant describes it and links it to threat context.
- **Incident reports** – Generate ISO 27001–style incident reports from a situation description.
- **Log analysis** – Collect and analyze logs (e.g. Windows Event Log) with LLM summarization.
- **Speech-to-text (STT)** – Voice input via Groq/OpenAI Whisper–compatible API.
- **File scan** – Heuristic malware scan (suspicious vs likely clean) for uploaded files.
- **Index explorer** – Search the RAG index and inspect retrieved chunks and metrics (Hit@k, MRR).

---

## Project structure

```
├── backend/           # FastAPI app (RAG, LLM, STT, file scan, logs)
│   ├── main.py
│   └── requirements.txt
├── frontend/           # React + Vite + TypeScript UI
├── rag_index/          # FAISS index + chunks + metadata (pre-built)
├── kaggle-rag-indexing.ipynb   # Build/rebuild RAG index (Kaggle or local)
├── collect_logs.ps1    # PowerShell script to collect Windows logs
└── README.md
```

---

## Quick start

### 1. Backend (FastAPI)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

Set environment variables (optional but recommended for full features):

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` or `GROQ_API_KEY` | LLM and STT (e.g. [Groq](https://console.groq.com/)) |
| `GROQ_BASE_URL` | Default: `https://api.groq.com/openai/v1` |
| `GROQ_MODEL` | Default: `llama-3.1-8b-instant` |
| `NVD_API_KEY` | Optional; enriches CVE details in incident reports ([NVD](https://nvd.nist.gov/developers/request-an-api-key)) |

Run the API (from the **project root**):

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

The backend expects the `rag_index/` folder (with `faiss.index`, `chunks.json`, `metadata.json`, `config.json`) next to the `backend/` folder.

### 2. Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

By default the UI talks to `http://localhost:8000`. Override with `VITE_API_BASE_URL` if needed.

### 3. Open the app

- Frontend: [http://localhost:5173](http://localhost:5173)
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Building the RAG index (optional)

The repo includes a pre-built `rag_index/` (FAISS + MiniLM-L3, 5.5k chunks). To rebuild or customize:

- **Kaggle:** Use `kaggle-rag-indexing.ipynb` with the NVD CVE/CPE and MITRE ATT&CK datasets. Set `GROQ_API_KEY` in Kaggle Secrets if you run LLM cells.
- **Local:** Run the notebook locally (or adapt the script) and point `CONFIG["index_dir"]` to your project’s `rag_index/` folder. Copy the generated files into `rag_index/`.

---

## API overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/ask` | RAG Q&A (with optional guided/non-technical mode) |
| POST | `/vision-rag` | Image + RAG analysis |
| POST | `/incident-report` | Generate incident report from description |
| POST | `/collect-logs` | Collect logs (e.g. Windows) |
| POST | `/analyze-logs` | Analyze collected logs with LLM |
| POST | `/stt` | Speech-to-text (audio file) |
| POST | `/scan-file` | Heuristic malware scan of uploaded file |

Request/response schemas are available at `/docs` (Swagger UI).

---

## Tech stack

- **Backend:** FastAPI, FAISS, sentence-transformers, rank-bm25, OpenAI-compatible client (Groq/OpenAI)
- **Frontend:** React 19, TypeScript, Vite
- **RAG:** NVD CVE/CPE, MITRE ATT&CK; chunking + dense (FAISS) retrieval; optional hybrid with BM25 in the notebook

---

## License

MIT.
