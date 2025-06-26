from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from uuid import uuid4
from typing import Optional, List, Dict, Tuple, Any
import datetime
import os
import openai
from dotenv import load_dotenv
import json
from app.utils.text_extraction import build_section_extraction_prompt, extract_section_text, SectionExtractionError, extract_text_from_pdf
from app.utils.question_generation import build_question_prompt
from app.question_pipeline import generate_question_set


# Load environment variables from .env
load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')

# Initialize the OpenAI client
client = openai.OpenAI(api_key=api_key)

app = FastAPI()

# In-memory store for now (replace with DB later)
DOCUMENTS = {}
QUESTIONS = {}

# Pydantic models
class DocumentInput(BaseModel):
    title: Optional[str] = None
    raw_text: str

class DocumentResponse(BaseModel):
    document_id: str
    status: str  # 'processing', 'ready', or 'error'

class Section(BaseModel):
    title: str
    first_sentence: str
    text: Optional[str] = None
    sub_sections: Optional[List[Dict]] = []

    class Config:
        from_attributes = True

Section.update_forward_refs()

class SectionDetectionResponse(BaseModel):
    sections: List[Section]
    learning_objectives: Dict[str, str]


class Question(BaseModel):
    question_text: str = Field(..., description="The question prompt")
    options: List[str] = Field(
        ..., 
        min_items=4, 
        max_items=4, 
        description="Exactly four answer choices"
    )
    correct_index: int = Field(
        ..., 
        ge=0, 
        le=3, 
        description="Index (0–3) of the correct option"
    )
    explanation: str = Field(..., description="One-sentence rationale for the correct answer")
    difficulty_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="0.0 = very easy … 1.0 = very hard"
    )
    concept_tags: List[str] = Field(
        ..., 
        min_items=1, 
        max_items=3, 
        description="1–3 topic tags for the question"
    )
    salience: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="0.0 = peripheral … 1.0 = core concept"
    )
    directness: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="0.0 = requires inference … 1.0 = stated literally"
    )

class QuestionGenerationRequest(BaseModel):
    section_text: str
    section_title: str
    num_questions: int = 2  # optional, default to 2
    learning_objectives: List[str] = []

class DocumentFullView(BaseModel):
    document_id: str
    title: Optional[str]
    raw_text: str
    status: str
    sections: Optional[List[Section]]
    learning_objectives: Optional[Dict[str, str]]


# Routes
@app.post("/documents/", response_model=DocumentResponse)
async def upload_document(doc: DocumentInput):
    # Simulate storing and start processing
    doc_id = str(uuid4())
    DOCUMENTS[doc_id] = {
        "title": doc.title,
        "raw_text": doc.raw_text,
        "upload_time": datetime.datetime.now().isoformat(),
        "status": "processing",
        "sections": None,
        "learning_objectives": None,
    }
    
    return {"document_id": doc_id, "status": "processing"}

@app.post("/documents/upload-file", response_model=DocumentResponse)
async def upload_document_file(file: UploadFile = File(...)):
    # Check file type
    if file.content_type == "application/pdf":
        raw_text = extract_text_from_pdf(file.file)
    elif file.content_type == "text/plain":
        raw_bytes = await file.read()
        raw_text = raw_bytes.decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    doc_id = str(uuid4())
    DOCUMENTS[doc_id] = {
        "title": file.filename,
        "raw_text": raw_text,
        "upload_time": datetime.datetime.now().isoformat(),
        "status": "processing",
        "sections": None,
        "learning_objectives": None,
    }

    return {"document_id": doc_id, "status": "processing"}


@app.post("/documents/{doc_id}/sections/detect", response_model=SectionDetectionResponse)
async def detect_sections(doc_id: str):
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found")

    raw_text = DOCUMENTS[doc_id]["raw_text"]

    # GPT prompt
    prompt = build_section_extraction_prompt(raw_text)
    # prompt = f"""
    # You are an AI Tutor tasked with identifying and labeling the sections of a given article. Please analyze the content and provide a hierarchical list of the sections and sub-sections that contain valuable, informative content. Exclude sections such as the abstract, references, and appendix.

    # For each section and sub-section, include the first 15 words of the section.

    # Also, generate a list of 3–5 key learning objectives a user should achieve after reading.

    # Return only valid JSON in this format:
    # {{
    # "sections": [
    #     {{
    #     "title": "<Title>",
    #     "first_sentence": "<First 15 words of this section’s content>",
    #     "sub_sections": []
    #     }},
    #     ...
    # ],
    # "learning_objectives": {{
    #     "1": "<Objective 1>",
    #     "2": "<Objective 2>",
    #     ...
    # }}
    # }}

    # Each section in the "sections" list should have:
    #         - "title": A short, clear label summarizing the section's content.
    #         - "first_sentence": The first sentence or header of that section.
    #         - "sub_sections": A list of sub-sections, each containing the same keys as above.

    # Here is the article:
    # \"\"\"
    # {raw_text}
    # \"\"\"
    # """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        gpt_output = response.choices[0].message.content

        # Try to safely parse JSON
        data = json.loads(gpt_output)
        sections = data["sections"]

        # Add section text to each section
        try:
            sections_with_text = extract_section_text(raw_text, sections)
            for section in sections_with_text:
                section["questions"] = []
        except SectionExtractionError as e:
            raise HTTPException(status_code=422, detail=str(e))

        # Save enriched sections
        DOCUMENTS[doc_id]["sections"] = sections_with_text
        DOCUMENTS[doc_id]["learning_objectives"] = data["learning_objectives"]
        DOCUMENTS[doc_id]["status"] = "ready"

        return {
            "sections": sections_with_text,
            "learning_objectives": data["learning_objectives"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing GPT response: {e}")


@app.post("/sections/{section_id}/questions/generate", response_model=List[Question])
async def generate_questions(section_id: str, req: QuestionGenerationRequest):
    try:
        prompt = build_question_prompt(
            section_text=req.section_text,
            section_title=req.section_title,
            num_questions=req.num_questions,
            learning_objectives=req.learning_objectives
        )

        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        parsed = json.loads(response.choices[0].message.content)
        print(parsed)

        # Normalize into a list
        if isinstance(parsed, dict) and "questions" in parsed:
            questions = parsed["questions"]
        elif isinstance(parsed, list):
            questions = parsed
        else:
            # single question object → wrap in a list
            questions = [parsed]

        # Store and return
        QUESTIONS[section_id] = questions
        return questions

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question generation failed: {e}")


@app.get("/documents/{doc_id}", response_model=DocumentFullView)
async def get_document(doc_id: str):
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found")

    doc = DOCUMENTS[doc_id]

    return {
        "document_id": doc_id,
        "title": doc.get("title"),
        "raw_text": doc["raw_text"],
        "status": doc.get("status", "processing"),
        "sections": doc.get("sections"),
        "learning_objectives": doc.get("learning_objectives"),
    }

@app.get("/documents/{doc_id}/questions", response_model=List[Question])
async def get_document_questions(doc_id: str):
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found")

    doc = DOCUMENTS[doc_id]
    sections = doc.get("sections", [])
    all_questions = []

    def collect_questions(section_list):
        for section in section_list:
            section_id = section["title"]  # Using title as stand-in ID
            if section_id in QUESTIONS:
                all_questions.extend(QUESTIONS[section_id])
            if "sub_sections" in section:
                collect_questions(section["sub_sections"])

    collect_questions(sections)

    return all_questions

import httpx

@app.post("/documents/{doc_id}/questions/generate-all")
async def generate_questions_for_all_sections(doc_id: str):
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found")

    doc = DOCUMENTS[doc_id]
    top_sections = doc.get("sections", [])
    global_learning_objectives = doc.get("learning_objectives", {})

    # Recursively collect only leaf sections (no sub_sections)
    def get_leaf_sections(sections: List[Dict], parent_index="") -> List[Tuple[str, Dict]]:
        leaf_sections = []
        for i, sec in enumerate(sections):
            section_id = f"{doc_id}_sec{parent_index}{i}"
            sub_secs = sec.get("sub_sections", [])
            if sub_secs:
                leaf_sections.extend(get_leaf_sections(sub_secs, parent_index=f"{parent_index}{i}_"))
            else:
                leaf_sections.append((section_id, sec))
        return leaf_sections

    leaf_sections = get_leaf_sections(top_sections)
    results: Dict[str, Any] = {}

    for section_id, section in leaf_sections:
        try:
            section_text = section.get("text", "").strip()
            if not section_text:
                results[section_id] = {"error": "No section text found"}
                continue

            local_learning_objectives = section.get("learning_objectives", [])
            questions = generate_question_set(
                section_text=section_text,
                num_questions=2,
                local_learning_objectives=local_learning_objectives,
                learning_objectives=global_learning_objectives
            )

            QUESTIONS[section_id] = questions
            results[section_id] = questions

        except Exception as e:
            results[section_id] = {"error": str(e)}

    return results


# @app.post("/documents/{doc_id}/questions/generate-all")
# async def generate_questions_for_all_sections(doc_id: str):
#     if doc_id not in DOCUMENTS:
#         raise HTTPException(status_code=404, detail="Document not found")

#     doc = DOCUMENTS[doc_id]
#     top_sections = doc.get("sections", [])
#     learning_objectives = list(doc.get("learning_objectives", {}).values())

#     # 1. Flatten the nested section tree into a single list
#     def flatten_sections(secs: list[dict]) -> list[dict]:
#         flat = []
#         for sec in secs:
#             flat.append(sec)
#             # Recurse into sub_sections if present
#             if sec.get("sub_sections"):
#                 flat.extend(flatten_sections(sec["sub_sections"]))
#         return flat

#     all_sections = flatten_sections(top_sections)

#     results: dict[str, any] = {}
#     async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as client:
#         for idx, section in enumerate(all_sections):
#             # Create a unique section_id for storage
#             section_id = f"{doc_id}_section_{idx}"
#             req_body = {
#                 "section_text": section.get("text", ""),
#                 "section_title": section["title"],
#                 "num_questions": 2,
#                 "learning_objectives": learning_objectives
#             }

#             response = await client.post(
#                 f"/sections/{section_id}/questions/generate",
#                 json=req_body
#             )

#             if response.status_code != 200:
#                 results[section_id] = {"error": response.text}
#             else:
#                 results[section_id] = response.json()

#     return results