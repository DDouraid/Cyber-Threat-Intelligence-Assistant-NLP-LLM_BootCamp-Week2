import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import subprocess
import sys
import re
import io

import faiss  # type: ignore
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer  # type: ignore
from openai import OpenAI
import requests


# --- Paths & index loading ----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "rag_index"

config_path = INDEX_DIR / "config.json"
chunks_path = INDEX_DIR / "chunks.json"
meta_path = INDEX_DIR / "metadata.json"
faiss_path = INDEX_DIR / "faiss.index"

if not (config_path.exists() and chunks_path.exists() and meta_path.exists() and faiss_path.exists()):
    raise RuntimeError(f"rag_index folder is incomplete under {INDEX_DIR}")

with config_path.open("r", encoding="utf-8") as f:
    RAG_CONFIG: Dict[str, Any] = json.load(f)

with chunks_path.open("r", encoding="utf-8") as f:
    CHUNK_TEXTS: List[str] = json.load(f)

with meta_path.open("r", encoding="utf-8") as f:
    CHUNK_META: List[Dict[str, Any]] = json.load(f)

INDEX = faiss.read_index(str(faiss_path))

EMBEDDING_MODEL_NAME = RAG_CONFIG.get(
    "embedding_model", "sentence-transformers/paraphrase-MiniLM-L3-v2"
)
DEFAULT_TOP_K: int = int(RAG_CONFIG.get("top_k", 5))

EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)


def _search_dense(query: str, top_k: int) -> List[Dict[str, Any]]:
    """FAISS dense search over precomputed chunk embeddings."""
    if not CHUNK_TEXTS:
        return []

    q_emb = EMBEDDING_MODEL.encode([query])
    q_emb = np.array(q_emb, dtype=np.float32)
    faiss.normalize_L2(q_emb)

    k = min(top_k, INDEX.ntotal)
    scores, indices = INDEX.search(q_emb, k)

    results: List[Dict[str, Any]] = []
    for i, s in zip(indices[0], scores[0]):
        idx = int(i)
        if idx < 0 or idx >= len(CHUNK_TEXTS):
            continue
        meta = CHUNK_META[idx] if idx < len(CHUNK_META) else {}
        results.append(
            {
                "idx": idx,
                "chunk": CHUNK_TEXTS[idx],
                "score": float(s),
                "source": meta.get("source", ""),
                "doc_id": meta.get("doc_id", ""),
                "title": meta.get("title", ""),
            }
        )
    return results


# --- RAG prompt template (from notebook) --------------------------------------

RAG_SYSTEM_PROMPT = (
    "You are a cybersecurity threat intelligence assistant. "
    "Answer only from the provided context. If the context does not contain enough information, "
    "say so. Be concise and cite CVE IDs or technique names when relevant. "
    "You may see MITRE ATT&CK techniques under ‘MITRE ATT&CK techniques’. "
    "When mapping attacks, prefer these techniques and do not invent techniques that are not in the context."
)

RAG_USER_TEMPLATE = """Context (from threat intelligence documents):

{context}

Question: {question}

Answer briefly and based only on the context above:"""


RAG_USER_TEMPLATE_GUIDED = """Context (from threat intelligence documents):

{context}

Question: {question}

We will go step by step.
In THIS response, explain ONLY **Step 1** in clear, non-technical language.
Do NOT describe later steps yet.
End by asking the user if they want to continue to the next step."""


GUIDED_NONTECH_PROMPT = (
    "You are helping a non-technical user in GUIDED MODE.\n"
    "Follow these rules strictly:\n"
    "1. Do NOT give the full solution at once.\n"
    "2. Break the solution into very small, simple steps.\n"
    "3. In each reply, explain only the NEXT step in plain language.\n"
    "4. In THIS response, output ONLY **Step 1** of the plan.\n"
    "5. Do NOT mention or summarize later steps yet.\n"
    "6. Avoid jargon; if you must use a technical term, explain it briefly.\n"
    "7. End every reply with a short question such as "
    "\"Does this make sense? If yes, type 'next' and I will give you the next step.\"\n"
)


def build_rag_prompt(
    context: str,
    question: str,
    max_context_chars: int = 6000,
    guided: bool = False,
) -> str:
    """Build the user prompt for RAG: context + question. Truncate context if needed."""
    context_trimmed = context[:max_context_chars] if len(context) > max_context_chars else context
    template = RAG_USER_TEMPLATE_GUIDED if guided else RAG_USER_TEMPLATE
    return template.format(context=context_trimmed, question=question)


# --- LLM client (Groq via OpenAI-compatible API, or any OpenAI-compatible host) ---

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY") or ""
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL") or os.environ.get(
    "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
)
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
VISION_MODEL = os.environ.get(
    "VISION_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct"
)

# NVD API key for enriching CVE details in reports (https://nvd.nist.gov/developers/request-an-api-key)
NVD_API_KEY = os.environ.get("NVD_API_KEY", "")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-large-v3")

LLM_CLIENT: Optional[OpenAI] = None
if OPENAI_API_KEY:
    LLM_CLIENT = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def _run_rag_llm(question: str, top_k: int, guided: bool = False) -> Dict[str, Any]:
    """Run dense retrieval + LLM. If no LLM key is configured, return only retrieved context."""
    results = _search_dense(question, top_k)
    context = "\n\n---\n\n".join(r["chunk"] for r in results)
    user_prompt = build_rag_prompt(context, question, guided=guided)

    if not LLM_CLIENT:
        return {
            "answer": (
                "RAG backend is configured and retrieval is working, but no LLM API key "
                "is set (OPENAI_API_KEY or GROQ_API_KEY). Here are the top retrieved chunks:\n\n"
                + context[:1500]
            ),
            "context_chunks": results,
            "used_top_k": top_k,
            "llm_provider": None,
        }

    system_prompt = RAG_SYSTEM_PROMPT
    if guided:
        system_prompt = (
            RAG_SYSTEM_PROMPT
            + " You are in GUIDED MODE for a non-technical user. "
            "Answer in very small, simple steps and, in this response, provide ONLY Step 1."
        )

    try:
        resp = LLM_CLIENT.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=500,
            temperature=0.4,
            top_p=0.9,
        )
        answer = resp.choices[0].message.content

        # Optional second LLM pass: simplify into step‑by‑step guidance
        # for non‑technical guided mode, while still using the RAG answer
        # as the source content.
        if guided and answer:
            try:
                coach_resp = LLM_CLIENT.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a friendly coach for a non‑technical user. "
                                "Rewrite answers into very simple, step‑by‑step guidance."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Here is an assistant answer based on cybersecurity context:\n\n"
                                f"{answer}\n\n"
                                "Rewrite this so that you ONLY describe the FIRST step the user "
                                "should take right now, in clear, everyday language. "
                                "Do not list later steps or summarize the whole plan. "
                                "End with a short question like: "
                                "\"Does this make sense? If yes, type 'next' and I will guide you to the next step.\""
                            ),
                        },
                    ],
                    max_tokens=350,
                    temperature=0.3,
                    top_p=0.9,
                )
                guided_answer = coach_resp.choices[0].message.content
                if guided_answer:
                    answer = guided_answer
            except Exception:
                # If the second pass fails, just fall back to the original answer.
                pass
    except Exception as e:  # pragma: no cover - defensive path
        answer = (
            f"Error while calling the LLM backend: {e}. "
            "Here are the top retrieved chunks so you can still inspect context:\n\n"
            + context[:1500]
        )

    return {
        "answer": answer,
        "context_chunks": results,
        "used_top_k": top_k,
        "llm_provider": OPENAI_BASE_URL,
    }


def _run_vision_rag(
    image_b64: str,
    mime_type: str,
    question: Optional[str],
    top_k: int,
) -> Dict[str, Any]:
    """
    Image → description (vision model) → dense RAG over index → RAG answer.
    Mirrors the image test cell from the notebook, adapted for an API.
    """
    if not LLM_CLIENT:
        return {
            "description": "",
            "rag_answer": (
                "Vision RAG is not available because no LLM API key is configured "
                "(OPENAI_API_KEY or GROQ_API_KEY)."
            ),
            "context_chunks": [],
            "used_top_k": top_k,
            "llm_provider": None,
        }

    # Accept either raw base64 or full data URL; always send a proper data URL to the LLM.
    data_url = image_b64.strip()
    if not data_url.startswith("data:"):
        mt = mime_type or "image/png"
        data_url = f"data:{mt};base64,{data_url}"

    # 1) Describe the image
    img_system = "You are a helpful cybersecurity assistant."
    img_prompt = (
        "Describe this image briefly, focusing on any security or risk-related elements if present."
    )

    try:
        img_resp = LLM_CLIENT.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": img_system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": img_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ],
            max_tokens=250,
            temperature=0.4,
            top_p=0.9,
        )
        description = img_resp.choices[0].message.content
    except Exception as e:  # pragma: no cover
        return {
            "description": "",
            "rag_answer": f"Vision API error: {e}",
            "context_chunks": [],
            "used_top_k": top_k,
            "llm_provider": OPENAI_BASE_URL,
        }

    # 2) Dense retrieval based on the description
    results = _search_dense(description, top_k)
    context = "\n\n---\n\n".join(r["chunk"] for r in results)

    # 3) RAG answer using threat-intel index
    q = question or (
        "Given the description of the image and the following threat-intel context, "
        "describe likely threats shown in the image and give concrete prevention "
        "and mitigation advice."
    )
    rag_user_prompt = build_rag_prompt(
        context,
        f"Image description: {description}\n\n{q}",
        guided=False,
    )

    try:
        rag_resp = LLM_CLIENT.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": rag_user_prompt},
            ],
            max_tokens=400,
            temperature=0.4,
            top_p=0.9,
        )
        rag_answer = rag_resp.choices[0].message.content
    except Exception as e:  # pragma: no cover
        rag_answer = (
            f"RAG+vision API error: {e}. Here is the image description instead:\n\n"
            f"{description}"
        )

    return {
        "description": description,
        "rag_answer": rag_answer,
        "context_chunks": results,
        "used_top_k": top_k,
        "llm_provider": OPENAI_BASE_URL,
    }


def _run_incident_report(
    description: str,
    audience: Optional[str],
    top_k: int,
) -> Dict[str, Any]:
    """
    Build an ISO 27001-style incident report using the cyber threat index as context.
    Mirrors the structured report cell in the notebook.
    """
    if not LLM_CLIENT:
        return {
            "report": (
                "Incident report generation is not available because no LLM API key "
                "is configured (OPENAI_API_KEY or GROQ_API_KEY)."
            ),
            "context_chunks": [],
            "used_top_k": top_k,
            "llm_provider": None,
        }

    # Retrieve context about similar incidents / threats
    results = _search_dense(description, top_k)
    context = "\n\n---\n\n".join(r["chunk"] for r in results)

    style = "technical and detailed" if (audience or "").lower() == "technical" else "simple and non-technical"

    # --- NVD CVE enrichment ----------------------------------------------------
    nvd_context_parts: List[str] = []
    if NVD_API_KEY:
        # Extract unique CVE IDs from description + context
        text_for_cves = description + "\n" + context
        cve_ids = list({m.group(0) for m in re.finditer(r"CVE-\d{4}-\d{4,7}", text_for_cves)})

        def _fetch_nvd_cve(cve_id: str) -> Optional[Dict[str, Any]]:
            try:
                resp = requests.get(
                    "https://services.nvd.nist.gov/rest/json/cves/2.0",
                    params={"cveId": cve_id, "apiKey": NVD_API_KEY},
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                return None
            vulns = data.get("vulnerabilities") or []
            return vulns[0] if vulns else None

        for cid in cve_ids[:5]:
            entry = _fetch_nvd_cve(cid)
            if not entry:
                continue
            cve = (entry.get("cve") or {})
            descriptions = cve.get("descriptions") or []
            desc = ""
            if descriptions:
                # Pick English description if available
                en = [d for d in descriptions if d.get("lang") == "en"]
                target = en[0] if en else descriptions[0]
                desc = target.get("value", "") or ""

            metrics = (cve.get("metrics") or {})
            cvss_text = ""
            # Try CVSS v3 then v2
            for key in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV3", "cvssMetricV2"]:
                arr = metrics.get(key)
                if not arr:
                    continue
                metric = arr[0]
                cvss_data = metric.get("cvssData") or {}
                base_score = cvss_data.get("baseScore")
                severity = metric.get("baseSeverity") or metric.get("cvssData", {}).get("baseSeverity")
                if base_score is not None or severity:
                    cvss_text = f"CVSS base score: {base_score}, severity: {severity}."
                    break

            summary_lines = [f"{cid}: {desc.strip()}"] if desc else [cid]
            if cvss_text:
                summary_lines.append(cvss_text)
            nvd_context_parts.append("\n".join(summary_lines))

    nvd_context = "\n\n".join(nvd_context_parts) if nvd_context_parts else ""

    iso_prompt = f"""
Using the context from the threat intelligence index and MITRE ATT&CK techniques,
generate an ISO 27001-style incident report for the following situation:

\"\"\"{description}\"\"\"

Context:
{context}

NVD CVE enrichment (authoritative details from the National Vulnerability Database, when available):
{nvd_context or 'No additional NVD details could be retrieved for the detected CVE IDs.'}

Structure the report using typical ISO 27001 incident management sections:
1. Incident identification and summary
2. Scope and impact (assets, data, users, business processes)
3. Cause and attack description (map to relevant MITRE ATT&CK techniques where possible)
4. Containment actions taken / recommended
5. Eradication and recovery steps
6. Lessons learned and preventive controls (policies, training, technical controls)
7. References to any relevant CVEs or incidents from the context

Write the report in a {style} way appropriate for the intended audience.
Base everything ONLY on the retrieved context and MITRE ATT&CK information. If
something is not supported by the context, say that explicitly instead of guessing.
"""

    try:
        resp = LLM_CLIENT.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": iso_prompt},
            ],
            max_tokens=900,
            temperature=0.4,
            top_p=0.9,
        )
        report = resp.choices[0].message.content
    except Exception as e:  # pragma: no cover
        report = f"Incident report generation error: {e}"

    return {
        "report": report,
        "context_chunks": results,
        "used_top_k": top_k,
        "llm_provider": OPENAI_BASE_URL,
    }


def _run_collect_logs() -> Dict[str, Any]:
    """
    Trigger the Windows PowerShell log collection script on the local machine.
    This mirrors the manual commands:
      Set-ExecutionPolicy Bypass -Scope Process
      .\\collect_logs.ps1

    The script itself handles elevation (RunAs) when needed.
    """
    if sys.platform != "win32":
        return {
            "ok": False,
            "message": "Log collection is only supported on Windows.",
            "json_folder": None,
            "evtx_folder": None,
            "return_code": None,
        }

    script_path = PROJECT_ROOT / "collect_logs.ps1"
    if not script_path.exists():
        return {
            "ok": False,
            "message": f"collect_logs.ps1 not found at {script_path}",
            "json_folder": None,
            "evtx_folder": None,
            "return_code": None,
        }

    # Mirrors Set-ExecutionPolicy Bypass -Scope Process; no separate step needed.
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "message": "Log collection script timed out after 600 seconds.",
            "json_folder": None,
            "evtx_folder": None,
            "return_code": None,
        }
    except Exception as e:  # pragma: no cover
        return {
            "ok": False,
            "message": f"Error while running log collection script: {e}",
            "json_folder": None,
            "evtx_folder": None,
            "return_code": None,
        }

    ok = completed.returncode == 0
    # The script always writes to Desktop\Cyber_SOC_Enterprise
    userprofile = os.environ.get("USERPROFILE", "")
    base = os.path.join(userprofile, "Desktop", "Cyber_SOC_Enterprise") if userprofile else None
    json_folder = os.path.join(base, "Json_Logs") if base else None
    evtx_folder = os.path.join(base, "EVTX_Logs") if base else None

    if ok:
        msg = (
            "Windows SOC log collection completed. "
            f"JSON logs are under: {json_folder} ; EVTX logs under: {evtx_folder}."
        )
    else:
        msg = (
            "Windows SOC log collection script exited with a non-zero code. "
            f"Stdout: {completed.stdout or '<empty>'}  Stderr: {completed.stderr or '<empty>'}"
        )

    return {
        "ok": ok,
        "message": msg,
        "json_folder": json_folder,
        "evtx_folder": evtx_folder,
        "return_code": int(completed.returncode),
    }


def _run_log_analysis(
    audience: Optional[str],
    top_k: int,
    max_chars: int,
) -> Dict[str, Any]:
    """
    Analyze collected Windows logs using the threat-intel RAG index + LLM.

    1. Read JSON logs from %USERPROFILE%\\Desktop\\Cyber_SOC_Enterprise\\Json_Logs.
    2. Concatenate and truncate to max_chars.
    3. Use dense retrieval over the threat index based on the log text.
    4. Ask the LLM to explain what the logs suggest and give prevention steps.
    """
    if not LLM_CLIENT:
        return {
            "answer": (
                "Log analysis is not available because no LLM API key is configured "
                "(OPENAI_API_KEY or GROQ_API_KEY)."
            ),
            "used_files": [],
            "used_top_k": top_k,
            "llm_provider": None,
        }

    userprofile = os.environ.get("USERPROFILE", "")
    if not userprofile:
        return {
            "answer": "Cannot locate USERPROFILE; log analysis is only supported on Windows user profiles.",
            "used_files": [],
            "used_top_k": top_k,
            "llm_provider": None,
        }

    base = os.path.join(userprofile, "Desktop", "Cyber_SOC_Enterprise", "Json_Logs")
    candidates = [
        os.path.join(base, "Security.json"),
        os.path.join(base, "System.json"),
        os.path.join(base, "Application.json"),
    ]

    texts: List[str] = []
    used_files: List[str] = []
    remaining = max(0, int(max_chars))

    for path in candidates:
        if remaining <= 0:
            break
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(remaining)
        except Exception:
            continue
        if not content.strip():
            continue
        texts.append(f"===== {os.path.basename(path)} =====\n{content}")
        used_files.append(path)
        remaining -= len(content)

    if not texts:
        return {
            "answer": (
                "I could not find any JSON logs under "
                f"{base}. Please run the log collection first and try again."
            ),
            "used_files": [],
            "used_top_k": top_k,
            "llm_provider": None,
        }

    log_text = "\n\n".join(texts)
    query_text = log_text[: min(2000, len(log_text))]

    results = _search_dense(query_text, top_k)
    context = "\n\n---\n\n".join(r["chunk"] for r in results)

    style = "technical and detailed" if (audience or "").lower() == "technical" else "simple and non-technical"

    user_prompt = f"""
You are analyzing recent Windows Security/System/Application logs collected from a user's PC.

[LOG SNIPPETS]
{log_text}

[THREAT-INTEL CONTEXT]
{context}

TASK:
1. Explain in {style} language what these logs might indicate (suspicious activity, failed logons, malware behavior, etc.).
2. Highlight any potential risks or red flags the user should care about.
3. Give concrete prevention and next steps for the user (what to do now, what to change to be safer).
4. Map to specific CVEs or MITRE ATT&CK techniques only when clearly supported by the context.
5. If something is uncertain or not in context, say so instead of guessing.
"""

    try:
        resp = LLM_CLIENT.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=900,
            temperature=0.4,
            top_p=0.9,
        )
        answer = resp.choices[0].message.content
    except Exception as e:  # pragma: no cover
        answer = f"Log analysis error: {e}"

    return {
        "answer": answer,
        "used_files": used_files,
        "used_top_k": top_k,
        "llm_provider": OPENAI_BASE_URL,
    }


# --- FastAPI app --------------------------------------------------------------


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    audience: Optional[str] = None  # "technical" | "nontechnical"


class RetrievedChunk(BaseModel):
    idx: int
    chunk: str
    score: float
    source: str = ""
    doc_id: str = ""
    title: str = ""


class AskResponse(BaseModel):
    answer: str
    used_top_k: int
    context_chunks: List[RetrievedChunk]
    llm_provider: Optional[str] = None


class VisionRagRequest(BaseModel):
    image_base64: str  # raw base64 or full data URL
    mime_type: Optional[str] = "image/png"
    question: Optional[str] = None
    top_k: Optional[int] = None


class VisionRagResponse(BaseModel):
    description: str
    rag_answer: str
    used_top_k: int
    context_chunks: List[RetrievedChunk]
    llm_provider: Optional[str] = None


class IncidentReportRequest(BaseModel):
    description: str
    audience: Optional[str] = None  # "technical" | "non-technical"
    top_k: Optional[int] = None


class IncidentReportResponse(BaseModel):
    report: str
    used_top_k: int
    context_chunks: List[RetrievedChunk]
    llm_provider: Optional[str] = None


class CollectLogsResponse(BaseModel):
    ok: bool
    message: str
    json_folder: Optional[str] = None
    evtx_folder: Optional[str] = None
    return_code: Optional[int] = None


class LogAnalysisRequest(BaseModel):
    audience: Optional[str] = None  # "technical" | "non-technical"
    top_k: Optional[int] = None
    max_chars: Optional[int] = 8000


class LogAnalysisResponse(BaseModel):
    answer: str
    used_files: List[str]
    used_top_k: int
    llm_provider: Optional[str] = None


class SttResponse(BaseModel):
    text: str


class MalwareScanResponse(BaseModel):
    filename: str
    size_bytes: int
    verdict: str  # "suspicious" | "likely_clean"
    score: float  # 0.0 (clean) -> 1.0 (very suspicious)
    reasons: List[str]


def _simple_malware_heuristics(filename: str, content: bytes) -> Dict[str, Any]:
    """
    Very simple, rule-based heuristic checker.
    This is for educational/testing purposes only and does NOT execute the file.
    It just looks for suspicious patterns in the filename and content.
    """
    size = len(content)
    reasons: List[str] = []
    score = 0.0

    lower_name = filename.lower()
    suspicious_exts = [
        ".exe",
        ".bat",
        ".cmd",
        ".ps1",
        ".vbs",
        ".js",
        ".scr",
        ".dll",
        ".sys",
        ".jar",
        ".docm",
        ".xlsm",
        ".pptm",
    ]
    if any(lower_name.endswith(ext) for ext in suspicious_exts):
        reasons.append(f"Suspicious extension detected: {lower_name}")
        score += 0.4

    # Try to interpret as text for keyword checks
    try:
        text_sample = content[:50000].decode("utf-8", errors="ignore").lower()
    except Exception:
        text_sample = ""

    # Keywords often associated with malicious behavior (for detection only)
    keyword_rules = {
        "keylogger": 0.3,
        "ransom": 0.3,
        "encryptfile": 0.3,
        "decryptfile": 0.2,
        "createprocess": 0.2,
        "virtualalloc": 0.2,
        "createremotethread": 0.3,
        "powershell -enc": 0.3,
        "powershell.exe": 0.2,
        "mimikatz": 0.5,
        "credential theft": 0.3,
        # EICAR test string: harmless but should be flagged as a test "malware" file
        "eicar-standard-antivirus-test-file": 0.7,
    }
    for kw, kw_score in keyword_rules.items():
        if kw in text_sample:
            reasons.append(f"Suspicious keyword found in content: '{kw}'")
            score += kw_score

    # Large binaries with very little readable text can be suspicious,
    # but this is a very rough heuristic.
    if size > 200 * 1024:  # > 200 KB
        text_len = len(text_sample)
        if text_len > 0 and (text_len / max(size, 1)) < 0.1:
            reasons.append("File is relatively large with little readable text.")
            score += 0.2

    # Cap score to [0, 1]
    score = max(0.0, min(1.0, score))
    verdict = "suspicious" if score >= 0.5 else "likely_clean"

    if not reasons:
        reasons.append("No obviously suspicious patterns were detected by simple heuristics.")

    return {
        "filename": filename,
        "size_bytes": size,
        "verdict": verdict,
        "score": float(score),
        "reasons": reasons,
    }


class LogAnalysisRequest(BaseModel):
    audience: Optional[str] = None  # "technical" | "non-technical"
    top_k: Optional[int] = None
    max_chars: Optional[int] = 8000


class LogAnalysisResponse(BaseModel):
    answer: str
    used_files: List[str]
    used_top_k: int
    llm_provider: Optional[str] = None


app = FastAPI(title="Cyber Threat Intelligence RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default
        "http://127.0.0.1:5173",
        "http://localhost:5174",  # fallback port when 5173 is busy
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "index_ntotal": int(INDEX.ntotal),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "has_llm": bool(LLM_CLIENT),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    top_k = int(req.top_k or DEFAULT_TOP_K)
    question = req.question
    audience = (req.audience or "").lower()

    # In non-technical mode, automatically switch to guided, step-by-step answers.
    guided = audience == "nontechnical"
    if guided:
        question = GUIDED_NONTECH_PROMPT + "\n\nUser question:\n" + question

    raw = _run_rag_llm(question, top_k, guided=guided)
    return AskResponse(
        answer=raw["answer"],
        used_top_k=raw["used_top_k"],
        context_chunks=[RetrievedChunk(**c) for c in raw["context_chunks"]],
        llm_provider=raw.get("llm_provider"),
    )


@app.post("/vision-rag", response_model=VisionRagResponse)
def vision_rag(req: VisionRagRequest) -> VisionRagResponse:
    """
    Image test endpoint: accepts an image, runs vision description + RAG over the cyber index,
    and returns both the description and the RAG-grounded answer.
    """
    top_k = int(req.top_k or DEFAULT_TOP_K)
    raw = _run_vision_rag(
        image_b64=req.image_base64,
        mime_type=req.mime_type or "image/png",
        question=req.question,
        top_k=top_k,
    )
    return VisionRagResponse(
        description=raw["description"],
        rag_answer=raw["rag_answer"],
        used_top_k=raw["used_top_k"],
        context_chunks=[RetrievedChunk(**c) for c in raw["context_chunks"]],
        llm_provider=raw.get("llm_provider"),
    )


@app.post("/incident-report", response_model=IncidentReportResponse)
def incident_report(req: IncidentReportRequest) -> IncidentReportResponse:
    """
    Generate an ISO 27001-style incident report from a free-form description,
    grounded in the cyber threat intelligence index.
    """
    top_k = int(req.top_k or DEFAULT_TOP_K)
    raw = _run_incident_report(
        description=req.description,
        audience=req.audience,
        top_k=top_k,
    )
    return IncidentReportResponse(
        report=raw["report"],
        used_top_k=raw["used_top_k"],
        context_chunks=[RetrievedChunk(**c) for c in raw["context_chunks"]],
        llm_provider=raw.get("llm_provider"),
    )


@app.post("/collect-logs", response_model=CollectLogsResponse)
def collect_logs() -> CollectLogsResponse:
    """
    Run the local PowerShell log collection script on the user's Windows machine.
    The script creates JSON + EVTX logs under:
      %USERPROFILE%\\Desktop\\Cyber_SOC_Enterprise\\{Json_Logs,EVTX_Logs}

    IMPORTANT: For full Security log access, run the backend (uvicorn) in an
    elevated PowerShell session (Run as Administrator), otherwise the script
    may fail or collect partial data.
    """
    raw = _run_collect_logs()
    return CollectLogsResponse(
        ok=bool(raw["ok"]),
        message=str(raw["message"]),
        json_folder=raw.get("json_folder"),
        evtx_folder=raw.get("evtx_folder"),
        return_code=raw.get("return_code"),
    )


@app.post("/analyze-logs", response_model=LogAnalysisResponse)
def analyze_logs(req: LogAnalysisRequest) -> LogAnalysisResponse:
    """
    Analyze collected Windows logs with the threat-intel RAG index and LLM.
    Uses JSON logs from %USERPROFILE%\\Desktop\\Cyber_SOC_Enterprise\\Json_Logs.
    """
    top_k = int(req.top_k or DEFAULT_TOP_K)
    max_chars = int(req.max_chars or 8000)
    raw = _run_log_analysis(
        audience=req.audience,
        top_k=top_k,
        max_chars=max_chars,
    )
    return LogAnalysisResponse(
        answer=str(raw["answer"]),
        used_files=list(raw.get("used_files") or []),
        used_top_k=top_k,
        llm_provider=raw.get("llm_provider"),
    )


@app.post("/stt", response_model=SttResponse)
async def stt(file: UploadFile = File(...), language: Optional[str] = None) -> SttResponse:
    """
    Speech-to-text via Groq/OpenAI Whisper-compatible endpoint.
    Expects an audio file (e.g. webm/ogg) and returns the transcribed text.
    """
    if not LLM_CLIENT:
        return SttResponse(
            text="Speech-to-text is not available because no LLM API key is configured (OPENAI_API_KEY or GROQ_API_KEY)."
        )

    data = await file.read()
    if not data:
        return SttResponse(text="")

    audio_file = io.BytesIO(data)
    audio_file.name = file.filename or "audio.webm"

    try:
        resp = LLM_CLIENT.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=audio_file,
            language=language or "en",
            response_format="text",
        )
        # For OpenAI 2.x+ with response_format="text", resp is a string
        if isinstance(resp, str):
            text = resp
        else:
            text = getattr(resp, "text", "") or ""
    except Exception as e:  # pragma: no cover
        text = f"STT error: {e}"

    return SttResponse(text=text)


@app.post("/scan-file", response_model=MalwareScanResponse)
async def scan_file(file: UploadFile = File(...)) -> MalwareScanResponse:
    """
    Simple, non-executing malware heuristic scan for an uploaded file.
    This is intended for testing your bot: it only does rule-based checks and
    does NOT guarantee real-world malware detection.
    """
    data = await file.read()
    if not data:
        result = {
            "filename": file.filename or "unknown",
            "size_bytes": 0,
            "verdict": "likely_clean",
            "score": 0.0,
            "reasons": ["Empty file (no content to analyze)."],
        }
    else:
        result = _simple_malware_heuristics(file.filename or "unknown", data)

    return MalwareScanResponse(**result)


@app.post("/collect-logs", response_model=CollectLogsResponse)
def collect_logs() -> CollectLogsResponse:
    """
    Run the local PowerShell log collection script on the user's Windows machine.
    The script creates JSON + EVTX logs under:
      %USERPROFILE%\Desktop\Cyber_SOC_Enterprise\{Json_Logs,EVTX_Logs}

    IMPORTANT: For full Security log access, run the backend (uvicorn) in an
    elevated PowerShell session (Run as Administrator), otherwise the script
    may fail or collect partial data.
    """
    raw = _run_collect_logs()
    return CollectLogsResponse(
        ok=bool(raw["ok"]),
        message=str(raw["message"]),
        json_folder=raw.get("json_folder"),
        evtx_folder=raw.get("evtx_folder"),
        return_code=raw.get("return_code"),
    )


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Cyber Threat Intelligence RAG backend is running."}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

