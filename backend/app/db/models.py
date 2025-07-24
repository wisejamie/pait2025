from sqlalchemy import (
    Column, String, Text, Integer, Float, ForeignKey, JSON, TIMESTAMP, func, Boolean, DateTime
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id           = Column(String, primary_key=True)     # your doc_id
    title        = Column(String, nullable=False)
    raw_text     = Column(Text, nullable=False)
    status       = Column(String, default="ready")
    created_at   = Column(TIMESTAMP, server_default=func.now())

    sections = relationship(
        "Section",
        back_populates="doc",
        cascade="all, delete-orphan",
        passive_deletes=True
    )
    learning_objectives = relationship(
        "LearningObjective",
        back_populates="doc",
        cascade="all, delete-orphan",
        passive_deletes=True
    )
    questions = relationship(
        "Question",
        back_populates="doc",
        cascade="all, delete-orphan",
        passive_deletes=True
    )


class Section(Base):
    __tablename__ = "sections"
    id            = Column(String, primary_key=True)      # your section id
    doc_id        = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    title         = Column(String)
    first_sentence = Column(Text)
    text          = Column(Text)
    sub_sections  = Column(JSON, default=[])            # keep nested structure if desired

    doc = relationship(
        "Document",
        back_populates="sections"
    )
    questions = relationship(
        "Question",
        back_populates="section",
        cascade="all, delete-orphan",
        passive_deletes=True
    )


class LearningObjective(Base):
    __tablename__ = "learning_objectives"
    id       = Column(Integer, primary_key=True, autoincrement=True)
    doc_id   = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    index    = Column(Integer)                         # 1,2,...
    objective = Column(Text)

    doc = relationship(
        "Document",
        back_populates="learning_objectives"
    )


class Question(Base):
    __tablename__ = "questions"
    id             = Column(Integer, primary_key=True, autoincrement=True)
    doc_id         = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    section_id     = Column(String, ForeignKey("sections.id", ondelete="CASCADE"), nullable=False)
    question_text  = Column(Text, nullable=False)
    correct_index  = Column(Integer)
    explanation    = Column(Text)
    difficulty_score = Column(Float)
    concept_tags   = Column(JSON, default=[])
    salience       = Column(Float)
    directness     = Column(Float)

    doc = relationship(
        "Document",
        back_populates="questions"
    )
    section = relationship(
        "Section",
        back_populates="questions"
    )
    options = relationship(
        "QuestionOption",
        back_populates="question",
        cascade="all, delete-orphan",
        passive_deletes=True
    )


class QuestionOption(Base):
    __tablename__ = "question_options"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    question_id = Column(Integer, ForeignKey("questions.id", ondelete="CASCADE"), nullable=False)
    idx         = Column(Integer)                      # 0,1,2,3
    text        = Column(Text)

    question = relationship(
        "Question",
        back_populates="options"
    )



class QuizSession(Base):
    __tablename__ = "quiz_sessions"
    id            = Column(String, primary_key=True)  # uuid4()
    doc_id        = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    start_time    = Column(DateTime, server_default=func.now())
    end_time      = Column(DateTime, nullable=True)

    # ── NEW ── track which question the user is on
    current_index = Column(Integer, nullable=False, default=0)

    session = relationship(
        "Document",
        backref=backref("quiz_sessions", cascade="all, delete-orphan", passive_deletes=True)
    )
    questions     = relationship(
        "QuizSessionQuestion",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True
    )

class QuizSessionQuestion(Base):
    __tablename__ = "quiz_session_questions"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    session_id    = Column(String, ForeignKey("quiz_sessions.id", ondelete="CASCADE"), nullable=False)
    question_id   = Column(Integer, ForeignKey("questions.id", ondelete="CASCADE"), nullable=False)
    ordinal       = Column(Integer, nullable=False)   # 0,1,2,...
    answered      = Column(Boolean, default=False)

    session       = relationship(
        "QuizSession",
        back_populates="questions"
    )
    question      = relationship(
        "Question"
    )
    response      = relationship(
        "QuizResponse",
        uselist=False,
        back_populates="session_question",
        cascade="all, delete-orphan",
        passive_deletes=True
    )

class QuizResponse(Base):
    __tablename__ = "quiz_responses"
    id                  = Column(Integer, primary_key=True, autoincrement=True)
    session_question_id = Column(
        Integer,
        ForeignKey("quiz_session_questions.id", ondelete="CASCADE"),
        unique=True
    )
    selected_index      = Column(Integer, nullable=False)
    correct             = Column(Boolean, nullable=False)

    session_question    = relationship(
        "QuizSessionQuestion",
        back_populates="response"
    )