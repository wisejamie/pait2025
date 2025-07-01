from pydantic import BaseModel, Field
from typing import List


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