# backend/scripts/prune_test_data.py
import os, sys
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from app.db.session import SessionLocal
from app.db.models import Document, Section, Question, QuestionOption
import pprint

def inspect_and_delete():
    db = SessionLocal()

    # 1) Inspect current rows
    print("\n--- DOCUMENTS BEFORE ---")
    docs = [{"id": d.id, "title": d.title} for d in db.query(Document).all()]
    pprint.pprint(docs)

    print("\n--- OPTIONS BEFORE ---")
    opts = [
        {"question_id": o.question_id, "text": o.text}
        for o in db.query(QuestionOption)
           .filter(QuestionOption.question_id == 12)
           .all()
    ]
    pprint.pprint(opts)

    # 2) Delete documents (and cascade their sections & questions)
    to_delete_docs = [
        "62698429-313a-4447-b262-a60eaf122440",
        "955cf1e2-cf1a-409d-beb7-d585bb0b557e",
    ]
    print(f"\nDeleting documents: {to_delete_docs}")
    # Remove related questions/options first
    # 2a) find all question IDs under these docs
    qids = [
        q.id
        for q in db.query(Question)
                 .filter(Question.section_id.like(tuple(f"{doc}_sec%" for doc in to_delete_docs)))
                 .all()
    ]
    if qids:
        print("Deleting options for questions:", qids)
        db.query(QuestionOption).filter(QuestionOption.question_id.in_(qids)).delete(synchronize_session=False)
        print("Deleting questions:", qids)
        db.query(Question).filter(Question.id.in_(qids)).delete(synchronize_session=False)

    # 2b) delete sections
    print("Deleting sections for those docs")
    db.query(Section).filter(Section.doc_id.in_(to_delete_docs)).delete(synchronize_session=False)

    # 2c) delete the documents
    db.query(Document).filter(Document.id.in_(to_delete_docs)).delete(synchronize_session=False)

    # 3) Delete the stray options on question 12
    print("\nDeleting stray options on question_id=12")
    db.query(QuestionOption) \
      .filter(QuestionOption.question_id == 12,
              QuestionOption.text.in_([
                  "3.3",
                  "GPT-4's performance on difficult texts is comparable to that of high school students, showing it has basic comprehension skills."
              ])) \
      .delete(synchronize_session=False)

    db.commit()

    # 4) Re-inspect
    print("\n--- DOCUMENTS AFTER ---")
    docs_after = [{"id": d.id, "title": d.title} for d in db.query(Document).all()]
    pprint.pprint(docs_after)

    print("\n--- OPTIONS FOR question_id=12 AFTER ---")
    opts_after = [
        {"question_id": o.question_id, "text": o.text}
        for o in db.query(QuestionOption)
                   .filter(QuestionOption.question_id == 12)
                   .all()
    ]
    pprint.pprint(opts_after)

    db.close()

if __name__ == "__main__":
    inspect_and_delete()
