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
from app.utils.question_pipeline import generate_question_set


# Load environment variables from .env
load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')

# Initialize the OpenAI client
client = openai.OpenAI(api_key=api_key)

app = FastAPI()

# In-memory store for now (replace with DB later)
DOCUMENTS = {}
QUESTIONS = {}
QUIZ_SESSIONS = {}

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


# Quiz Session
class QuizSessionCreateRequest(BaseModel):
    document_id: str
    num_questions: Optional[int] = 10

class AnswerSubmission(BaseModel):
    question_id: str
    selected_index: int

import random
from datetime import datetime
@app.post("/quiz-sessions/")
async def create_quiz_session(req: QuizSessionCreateRequest):
    doc_id = req.document_id
    num_questions = req.num_questions

    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found")

    # Gather all questions for this document
    all_question_ids = []
    for section_id in QUESTIONS:
        if section_id.startswith(doc_id):
            for idx, q in enumerate(QUESTIONS[section_id]):
                question_id = f"{section_id}_q{idx}"
                all_question_ids.append((question_id, section_id, q))

    if not all_question_ids:
        raise HTTPException(status_code=400, detail="No questions found for this document")

    # Randomly select up to num_questions
    selected = random.sample(all_question_ids, min(num_questions, len(all_question_ids)))
    selected_ids = [qid for qid, _, _ in selected]

    # Store the session
    session_id = str(uuid4())
    QUIZ_SESSIONS[session_id] = {
        "document_id": doc_id,
        "question_refs": selected,  # (question_id, section_id, question_obj)
        "question_ids": selected_ids,
        "responses": [],
        "current_index": 0,
        "start_time": datetime.utcnow().isoformat(),
        "end_time": None
    }

    return {
        "session_id": session_id,
        "total_questions": len(selected_ids),
        "status": "active"
    }

@app.get("/quiz-sessions/{session_id}/next")
async def get_next_question(session_id: str):
    if session_id not in QUIZ_SESSIONS:
        raise HTTPException(status_code=404, detail="Quiz session not found")

    session = QUIZ_SESSIONS[session_id]
    index = session["current_index"]
    total = len(session["question_refs"])

    if index >= total:
        return {"message": "Quiz completed", "finished": True}

    question_id, section_id, question_obj = session["question_refs"][index]

    # Return question data (hide answer metadata)
    return {
        "question_id": question_id,
        "index": index,
        "total": total,
        "question": {
            "question_text": question_obj["question_text"],
            "options": question_obj["options"]
        }
    }

@app.post("/quiz-sessions/{session_id}/answer")
async def submit_answer(session_id: str, submission: AnswerSubmission):
    if session_id not in QUIZ_SESSIONS:
        raise HTTPException(status_code=404, detail="Quiz session not found")

    session = QUIZ_SESSIONS[session_id]
    index = session["current_index"]
    total = len(session["question_refs"])

    if index >= total:
        raise HTTPException(status_code=400, detail="Quiz already completed")

    current_qid, section_id, question_obj = session["question_refs"][index]

    if submission.question_id != current_qid:
        raise HTTPException(status_code=400, detail="Submitted question does not match current question")

    is_correct = submission.selected_index == question_obj["correct_index"]

    session["responses"].append({
        "question_id": current_qid,
        "selected_index": submission.selected_index,
        "correct": is_correct
    })

    session["current_index"] += 1

    return {
        "correct": is_correct,
        "correct_index": question_obj["correct_index"],
        "explanation": question_obj.get("explanation", ""),
        "next_index": session["current_index"],
        "completed": session["current_index"] >= total
    }

@app.get("/quiz-sessions/{session_id}/summary")
async def get_quiz_summary(session_id: str):
    if session_id not in QUIZ_SESSIONS:
        raise HTTPException(status_code=404, detail="Quiz session not found")

    session = QUIZ_SESSIONS[session_id]
    total = len(session["question_refs"])
    responses = session["responses"]

    if len(responses) < total:
        return {"message": "Quiz not yet complete", "finished": False}

    num_correct = sum(1 for r in responses if r["correct"])
    num_incorrect = total - num_correct
    score_pct = round((num_correct / total) * 100, 2)

    missed_questions = []
    for i, response in enumerate(responses):
        if not response["correct"]:
            qid, section_id, qobj = session["question_refs"][i]
            missed_questions.append({
                "question_id": qid,
                "question_text": qobj["question_text"],
                "options": qobj["options"],
                "selected_index": response["selected_index"],
                "correct_index": qobj["correct_index"],
                "explanation": qobj.get("explanation", "")
            })

    return {
        "total_questions": total,
        "correct": num_correct,
        "incorrect": num_incorrect,
        "score_percent": score_pct,
        "missed_questions": missed_questions,
        "finished": True
    }




# FOR TESTING PURPOSES:
@app.on_event("startup")
async def load_test_data():
    doc_id = "test-doc"

    # Add fake document
    DOCUMENTS[doc_id] = {
        "sections": [],
        "learning_objectives": {
            "1": "Understand symbolic reasoning",
            "2": "Recognize patterns in AI models"
        }
    }

    # Add fake questions for this doc
    QUESTIONS[f"{doc_id}_sec0"] = [
        {
            "question_text": "What is symbolic reasoning?",
            "options": ["Rule-based logic", "Random guessing", "Emotional response", "Genetic programming"],
            "correct_index": 0,
            "explanation": "Symbolic reasoning uses logical rules and symbols to derive conclusions.",
            "difficulty_score": 0.3,
            "concept_tags": ["symbolic reasoning"],
            "salience": 0.9,
            "directness": 1.0
        },
        {
            "question_text": "Which of the following best defines pattern recognition in AI?",
            "options": ["Identifying emotional responses", "Recognizing trends in data", "Following hard-coded rules", "Generating random outcomes"],
            "correct_index": 1,
            "explanation": "Pattern recognition refers to identifying trends or structures in data inputs.",
            "difficulty_score": 0.4,
            "concept_tags": ["pattern recognition"],
            "salience": 0.8,
            "directness": 0.9
        }
    ]
