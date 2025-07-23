from pydantic import BaseModel, Field
from typing import Optional, List

class QuizSessionCreateRequest(BaseModel):
    document_id: str
    num_questions: Optional[int] = 10
    sections: Optional[List[str]] = None 

class AnswerSubmission(BaseModel):
    question_id: str
    selected_index: int

class SectionScore(BaseModel):
    section_id: str
    section_title: str
    correct: int
    incorrect: int
    total: int
    percent: float

class MissedQuestion(BaseModel):
    question_id: str
    question_text: str
    options: List[str]
    selected_index: int
    correct_index: int
    explanation: str

class QuizSummaryOut(BaseModel):
    document_id: str
    total_questions: int
    correct: int
    incorrect: int
    score_percent: float
    section_scores: List[SectionScore]
    missed_questions: List[MissedQuestion]
    finished: bool