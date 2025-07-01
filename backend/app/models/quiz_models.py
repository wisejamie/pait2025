from pydantic import BaseModel, Field
from typing import Optional

class QuizSessionCreateRequest(BaseModel):
    document_id: str
    num_questions: Optional[int] = 10

class AnswerSubmission(BaseModel):
    question_id: str
    selected_index: int