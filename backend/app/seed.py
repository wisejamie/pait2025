import pprint
from typing import List, Dict
from app.db.session import SessionLocal
from app.db.models import Document, Section, LearningObjective, Question, QuestionOption
from app.storage.memory import *
from sqlalchemy import text

# your helpers
def flatten_sections(sections: List[Dict]) -> List[Dict]:
    flat = []
    for section in sections:
        flat.append(section)
        if isinstance(section.get("sub_sections"), list):
            flat.extend(flatten_sections(section["sub_sections"]))
    return flat

def add_section_ids(sections: List[Dict], doc_id: str, parent_index: str = "") -> List[Dict]:
    for i, section in enumerate(sections):
        # assign id to this section
        section_id = f"{doc_id}_sec{parent_index}{i}"
        section["id"] = section_id

        # recurse into sub_sections
        print(f"[add_section_ids] assigned id={section_id} to section title={section.get('title')}")
        sub_secs = section.get("sub_sections") or []
        if isinstance(sub_secs, list) and sub_secs:
            next_parent = f"{parent_index}{i}_"
            add_section_ids(sub_secs, doc_id, parent_index=next_parent)

    return sections

def seed():
    db = SessionLocal()

    # ─── WIPE EXISTING DATA ──────────────────────────
    for tbl in (
        # "quiz_responses",
        # "quiz_session_questions",
        # "quiz_sessions",
        "question_options",
        "questions",
        "learning_objectives",
        "sections",
        "documents",
    ):
        db.execute(text(f"DELETE FROM {tbl}"))
    db.commit()

    for doc_id, doc in DOCUMENTS.items():
        print(f"\n[seed] Seeding document {doc_id!r} title={doc['title']!r}")
        d = Document(id=doc_id, title=doc["title"], raw_text=doc["raw_text"], status=doc["status"])
        db.add(d)

        # 1) Assign IDs
        tree = doc.get("sections", [])
        print("[seed] Original sections tree:")
        pprint.pprint(tree, indent=2)
        add_section_ids(tree, doc_id)

        print("[seed] After add_section_ids (nested):")
        pprint.pprint(tree, indent=2)

        # 2) Flatten
        flat = flatten_sections(tree)
        print("[seed] Flat list of sections to insert:")
        for sec in flat:
            print(f"   id={sec['id']} title={sec.get('title')!r}")

        # 3) Insert them
        for sec in flat:
            s = Section(
                id=sec["id"], doc_id=doc_id,
                title=sec.get("title",""), first_sentence=sec.get("first_sentence",""),
                text=sec.get("text",""), sub_sections=sec.get("sub_sections", [])
            )
            db.add(s)

        # 4) Insert learning objectives
        for idx, obj in doc.get("learning_objectives", {}).items():
            lo = LearningObjective(
                doc_id=doc_id,
                index=int(idx),
                objective=obj,
            )
            db.add(lo)


    for key, qs in QUESTIONS.items():
        doc_id, sec_id = key.split("_", 1)
        for q in qs:
            question = Question(
                doc_id=doc_id,
                section_id=f"{doc_id}_{sec_id}",
                question_text=q["question_text"],
                correct_index=q["correct_index"],
                explanation=q["explanation"],
                difficulty_score=q["difficulty_score"],
                concept_tags=q["concept_tags"],
                salience=q["salience"],
                directness=q["directness"],
            )
            db.add(question)
            db.flush()  # get question.id
            for idx, opt in enumerate(q["options"]):
                qo = QuestionOption(
                    question_id=question.id,
                    idx=idx,
                    text=opt,
                )
                db.add(qo)
    db.commit()
    db.close()

if __name__ == "__main__":
    seed()
