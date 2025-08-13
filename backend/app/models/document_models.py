from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal


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


class SectionDetectionResponse(BaseModel):
    sections: List[Section]
    learning_objectives: Dict[str, str]


class DocumentView(BaseModel):
    document_id: str
    title: str
    upload_time: str  # alternatively, use datetime


class DocumentFullView(BaseModel):
    document_id: str
    title: str
    sections: List[Section]
    learning_objectives: Dict[str, str]


class SummaryRequest(BaseModel):
    level: Literal["tldr", "short", "bullets", "simple"]

class SummaryResponse(BaseModel):
    summary: str

class TransformRequest(BaseModel):
    mode: Literal["simplify", "elaborate", "distill"]

class TransformResponse(BaseModel):
    transformedText: str

class ExplainRequest(BaseModel):
    section_text: str   # the full paragraph (or section) for context
    snippet: str        # the user-highlighted text

class AskRequest(BaseModel):
    question: str
    context: Literal["document", "section"]
    section_id: Optional[str] = None
    history: Optional[List[Dict]] = None
    max_tokens: Optional[int] = 800
    include_sources: Optional[bool] = False

class AskResponse(BaseModel):
    answer: str
    used_context: Literal["document", "section", "auto"]
    sources: Optional[List[Dict]] = None
    token_usage: Optional[Dict] = None
    warnings: Optional[List[str]] = None