from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uuid import uuid4
from typing import Optional, List, Dict
import datetime
import os
import openai
from dotenv import load_dotenv
import json
from app.utils.text_extraction import extract_section_text, SectionExtractionError
from app.utils.question_generation import build_question_prompt


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
        orm_mode = True

Section.update_forward_refs()

class SectionDetectionResponse(BaseModel):
    sections: List[Section]
    learning_objectives: Dict[str, str]


class Question(BaseModel):
    question_text: str
    options: List[str]
    correct_index: int
    explanation: str
    difficulty: Optional[str] = "medium"  # optional with default
    skill: Optional[str] = None
    tags: Optional[List[str]] = []

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
    # TODO: Call GPT to detect sections and store them
    
    return {"document_id": doc_id, "status": "processing"}

@app.post("/documents/{doc_id}/sections/detect", response_model=SectionDetectionResponse)
async def detect_sections(doc_id: str):
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found")

    raw_text = DOCUMENTS[doc_id]["raw_text"]

    # GPT prompt
    prompt = f"""
    You are an AI Tutor tasked with identifying and labeling the sections of a given article. Please analyze the content and provide a hierarchical list of the sections and sub-sections that contain valuable, informative content. Exclude sections such as the abstract, references, and appendix.

    For each section and sub-section, include the first sentence or header of the section.

    Also, generate a list of 3–5 key learning objectives a user should achieve after reading.

    Return only valid JSON in this format:
    {{
    "sections": [
        {{
        "title": "<Title>",
        "first_sentence": "<First Sentence>",
        "sub_sections": []
        }},
        ...
    ],
    "learning_objectives": {{
        "1": "<Objective 1>",
        "2": "<Objective 2>",
        ...
    }}
    }}

    Each section in the "sections" list should have:
            - "title": A short, clear label summarizing the section's content.
            - "first_sentence": The first sentence or header of that section.
            - "sub_sections": A list of sub-sections, each containing the same keys as above.

    Here is the article:
    \"\"\"
    {raw_text}
    \"\"\"
    """

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
    from app.utils.text_extraction import SectionExtractionError  # optional if reused

    try:
        prompt = build_question_prompt(
            section_text=req.section_text,
            section_title=req.section_title,
            num_questions=req.num_questions,
            learning_objectives=req.learning_objectives
        )

        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        gpt_output = response.choices[0].message.content
        parsed = json.loads(gpt_output)

        if isinstance(parsed, dict) and "questions" in parsed:
            questions = parsed["questions"]
        else:
            questions = parsed
        
        QUESTIONS[section_id] = questions

        # Optional: validate all entries conform to schema (let FastAPI do it)
        return questions

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")

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
