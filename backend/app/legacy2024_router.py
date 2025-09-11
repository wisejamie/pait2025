# ==== LEGACY 2024 ENDPOINTS (pure 2024 logic, FastAPI wrapper) ====

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from uuid import uuid4
from typing import Dict, Any, Optional
import io
import PyPDF2
import threading

# Import your unchanged 2024 logic (rename your old file to legacy_2024.py)
# These must be the IDENTICAL functions & prompts you used in 2024.
from app.legacy_2024 import (
    set_article_text as legacy_set_article_text,
    get_article_text as legacy_get_article_text,
    extract_sections as legacy_extract_sections,
    extract_section_text as legacy_extract_section_text,
)

legacy2024_router = APIRouter(prefix="/legacy2024", tags=["legacy-2024"])

# In-memory store so we don't depend on 2024's globals beyond a single call.
# We compute with the 2024 globals, read the result, then persist per-doc here.
DOCUMENTS_2024: Dict[str, Dict[str, Any]] = {}
DOC_LOCK = threading.Lock()


class TextIn(BaseModel):
    title: Optional[str] = None
    raw_text: str


def _extract_text_from_pdf_bytes(data: bytes) -> str:
    """Byte-for-byte replica of your 2024 PyPDF2 extraction."""
    with io.BytesIO(data) as stream:
        reader = PyPDF2.PdfReader(stream)
        text = ""
        for page in reader.pages:
            # EXACT behavior as 2024:
            page_text = page.extract_text() if page.extract_text() else ""
            text += page_text
        return text


def _compute_sections_with_2024_logic(raw_text: str) -> Dict[str, Any]:
    """
    Run the 2024 pipeline exactly:
      1) set_article_text(raw_text)  -> triggers 2024's section setup call
      2) extract_sections()          -> gets {"sections", "learning_objectives"}
      3) extract_section_text(...)   -> fuzzy-anchors and fills 'text' per section
    Return {"sections": [...], "learning_objectives": {...}}
    """
    # Guard the legacy module's globals during computation
    with DOC_LOCK:
        legacy_set_article_text(raw_text)
        # Call the same outline prompt/logic from 2024
        outline = legacy_extract_sections()
        sections = outline.get("sections", []) or []
        learning_objectives = outline.get("learning_objectives", {}) or {}

        # Use the EXACT same anchoring the 2024 code used
        anchored_sections = legacy_extract_section_text(legacy_get_article_text(), sections)

    return {
        "sections": anchored_sections,
        "learning_objectives": learning_objectives,
    }


@legacy2024_router.post("/documents/upload")
async def upload_pdf_2024(file: UploadFile = File(...)):
    """
    Upload a PDF and process it with the 2024 pipeline (PyPDF2 + 2024 prompts/anchoring).
    Returns a doc_id you can query for sections.
    """
    if file.content_type not in ("application/pdf", "application/x-pdf", "binary/octet-stream"):
        raise HTTPException(status_code=415, detail="Please upload a PDF.")

    data = await file.read()
    try:
        raw_text = _extract_text_from_pdf_bytes(data)
        if not raw_text.strip():
            raise ValueError("No extractable text (PDF may be a scan or image-only).")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {e}")

    doc_id = str(uuid4())
    DOCUMENTS_2024[doc_id] = {
        "title": file.filename or "Untitled",
        "raw_text": raw_text,
        "sections": None,
        "learning_objectives": None,
        "status": "uploaded",
    }
    return {"document_id": doc_id, "status": "uploaded"}


@legacy2024_router.post("/documents/text")
async def ingest_text_2024(payload: TextIn):
    """
    Provide raw text directly (no PDF) and process with the 2024 pipeline.
    """
    raw_text = (payload.raw_text or "").strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="raw_text is empty.")

    doc_id = str(uuid4())
    DOCUMENTS_2024[doc_id] = {
        "title": payload.title or "Untitled",
        "raw_text": raw_text,
        "sections": None,
        "learning_objectives": None,
        "status": "uploaded",
    }
    return {"document_id": doc_id, "status": "uploaded"}


@legacy2024_router.post("/documents/{doc_id}/sections/recompute")
def recompute_sections_2024(doc_id: str):
    """
    Force recompute sections using the 2024 logic.
    Useful if you change the 2024 prompts/thresholds and want to refresh.
    """
    doc = DOCUMENTS_2024.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Unknown document_id.")

    try:
        result = _compute_sections_with_2024_logic(doc["raw_text"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"2024 sectioning failed: {e}")

    doc["sections"] = result["sections"]
    doc["learning_objectives"] = result["learning_objectives"]
    doc["status"] = "sectioned"
    return {"document_id": doc_id, "status": "sectioned"}


@legacy2024_router.get("/documents/{doc_id}/sections")
def get_sections_2024(doc_id: str):
    """
    Get sections (with text spans) + learning objectives, computed via 2024 flow.
    If not yet computed, computes on first GET to match the old “lazy” behavior.
    """
    doc = DOCUMENTS_2024.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Unknown document_id.")

    if doc["sections"] is None:
        try:
            result = _compute_sections_with_2024_logic(doc["raw_text"])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"2024 sectioning failed: {e}")
        doc["sections"] = result["sections"]
        doc["learning_objectives"] = result["learning_objectives"]
        doc["status"] = "sectioned"

    return {
        "document_id": doc_id,
        "title": doc["title"],
        "sections": doc["sections"],
        "learning_objectives": doc["learning_objectives"],
        "status": doc["status"],
    }
