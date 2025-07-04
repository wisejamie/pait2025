from pydantic import BaseModel, Field
from typing import Optional, List

class QuizSessionCreateRequest(BaseModel):
    document_id: str
    num_questions: Optional[int] = 10
    sections: Optional[List[str]] = None 

class AnswerSubmission(BaseModel):
    question_id: str
    selected_index: int