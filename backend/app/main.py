import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from uuid import uuid4
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime
import os
import openai
from dotenv import load_dotenv
import json
from app.utils.text_extraction import build_section_extraction_prompt, extract_section_text, SectionExtractionError, extract_text_from_pdf
from app.utils.question_pipeline import generate_question_set
from app.models.document_models import *
from app.models.question_models import *
from app.models.quiz_models import *
from app.storage.memory import *


from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env
load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')

# Initialize the OpenAI client
client = openai.OpenAI(api_key=api_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Section.update_forward_refs()

# Routes
@app.post("/documents/", response_model=DocumentResponse)
async def upload_document(doc: DocumentInput):
    # Simulate storing and start processing
    doc_id = str(uuid4())
    DOCUMENTS[doc_id] = {
        "title": doc.title,
        "raw_text": doc.raw_text,
        "upload_time": datetime.now().isoformat(),
        "status": "processing",
        "sections": [],
        "learning_objectives": {},
    }
    
    return {"document_id": doc_id, "status": "processing"}


@app.get("/documents", response_model=List[DocumentView])
def list_documents():
    return [
        {
            "document_id": doc_id,
            "title": doc.get("title", "Untitled"),
            "upload_time": doc.get("upload_time", datetime.now().isoformat())
        }
        for doc_id, doc in DOCUMENTS.items()
    ]

@app.get("/documents/{doc_id}", response_model=DocumentFullView)
async def get_document(doc_id: str):
    doc = DOCUMENTS.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "document_id": doc_id,
        "title": doc.get("title", "Untitled"),
        "raw_text": doc.get("raw_text", ""),
        "status": doc.get("status", "ready"),
        "sections": doc.get("sections", []),
        "learning_objectives": doc.get("learning_objectives", {}),
    }

@app.get("/documents/{doc_id}/sections")
def get_sections(doc_id: str):
    doc = DOCUMENTS.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    sections = doc.get("sections", [])
    learning_objectives = doc.get("learning_objectives", {})

    # Ensure section IDs are assigned
    sections = add_section_ids(sections)

    return {
        "sections": sections,
        "learning_objectives": learning_objectives,
    }


@app.get("/sections/{section_id}")
def get_section(section_id: str):
    for doc in DOCUMENTS.values():
        sections = add_section_ids(doc.get("sections", []))
        flat_sections = flatten_sections(sections)
        for section in flat_sections:
            if section.get("id") == section_id:
                return section

    raise HTTPException(status_code=404, detail="Section not found")

def flatten_sections(sections):
    """Flatten a nested list of sections and their sub_sections into a single flat list."""
    flat = []

    for section in sections:
        flat.append(section)
        if "sub_sections" in section and isinstance(section["sub_sections"], list):
            flat.extend(flatten_sections(section["sub_sections"]))

    return flat


def add_section_ids(sections):
    for section in sections:
        if "id" not in section:
            section["id"] = str(uuid.uuid4())
        if "sub_sections" in section:
            add_section_ids(section["sub_sections"])
    return sections

@app.post("/documents/upload-file", response_model=DocumentResponse)
async def upload_document_file(file: UploadFile = File(...), title: str = Form(...)):
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
        "title": title,
        "raw_text": raw_text,
        "upload_time": datetime.now().isoformat(),
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
    doc = DOCUMENTS.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    sections = doc.get("sections", [])
    all_questions = []

    def collect_questions(section_list):
        for section in section_list:
            section_id = section.get("id")
            if section_id and section_id in QUESTIONS:
                all_questions.extend(QUESTIONS[section_id])
            if "sub_sections" in section:
                collect_questions(section["sub_sections"])

    collect_questions(sections)
    return all_questions

@app.post("/documents/{doc_id}/questions/generate-all")
async def generate_questions_for_all_sections(doc_id: str):
    doc = DOCUMENTS.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

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

import random
@app.post("/quiz-sessions/")
async def create_quiz_session(req: QuizSessionCreateRequest):
    doc_id = req.document_id
    num_questions = req.num_questions

    print(doc_id)
    print(DOCUMENTS)
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found")

    # Gather all questions for this document
    all_question_ids = []

    question_source = QUESTIONS.copy()

    for section_id, questions in question_source.items():
        if section_id.startswith(doc_id):
            for idx, q in enumerate(questions):
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
        "start_time": datetime.now().isoformat(),
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
        "document_id": "test-doc",
        "title": "Test Document on AI Comprehension",
        "raw_text": "This is a sample document about how AI models comprehend text. It contains one section with meaningful content.",
        "status": "ready",
        "sections": [
            {
            "title": "Introduction to AI Comprehension",
            "first_sentence": "This is a sample document about how AI models comprehend text.",
            "text": "This is a sample document about how AI models comprehend text. It contains one section with meaningful content.",
            "sub_sections": []
            }
        ],
        "learning_objectives": {
            "1": "Understand how AI models interpret language.",
            "2": "Identify evaluation metrics for AI comprehension."
        }
    }


    # Add fake questions for this doc
    QUESTIONS[f"{doc_id}_sec0"] = [
        {
            "question_text": "What is the main topic of the document?",
            "options": [
            "How AI models comprehend text",
            "History of artificial intelligence",
            "Syntax in natural languages",
            "Computer vision techniques"
            ],
            "correct_index": 0,
            "explanation": "The document is about how AI models comprehend text.",
            "difficulty_score": 0.2,
            "concept_tags": ["AI comprehension"],
            "salience": 0.8,
            "directness": 0.9
        },
        {
            "question_text": "Which of the following best describes the content of the section?",
            "options": [
            "Details computer vision models",
            "Introduces the topic of AI understanding of text",
            "Explains hardware requirements for AI",
            "Describes neural network architectures"
            ],
            "correct_index": 1,
            "explanation": "The section introduces how AI models comprehend language.",
            "difficulty_score": 0.3,
            "concept_tags": ["language understanding"],
            "salience": 0.7,
            "directness": 0.85
        }
    ]

