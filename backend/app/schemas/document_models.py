from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal


class DocumentInput(BaseModel):
    title: Optional[str] = None
    raw_text: str


class DocumentResponse(BaseModel):
    document_id: str
    status: str  # 'processing', 'ready', or 'error'

class Section(BaseModel):
    id: str                          # ← add this
    title: str
    first_sentence: str
    text: Optional[str] = None
    sub_sections: List["Section"] = []   # recursive nesting
    questions: List[Dict] = []      # if you include questions in this response

    class Config:
        from_attributes = True
        # allow recursive models
        orm_mode = True

# needed to allow Section to reference itself
Section.update_forward_refs()


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