import os, sys
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
from app.db.models import *
from app.db.session import SessionLocal
import sys
import pprint

def main():
    if len(sys.argv) < 2:
        print("Usage: prune_docs.py <doc_id1> [<doc_id2> ...]")
        sys.exit(1)

    # first argument onward are the document IDs to delete
    to_delete_docs = sys.argv[1:]
    print("🗑️  Will delete documents:", to_delete_docs)

    db = SessionLocal()

    # delete learning objectives, sections, and documents
    db.query(LearningObjective)\
      .filter(LearningObjective.doc_id.in_(to_delete_docs))\
      .delete(synchronize_session=False)

    db.query(Section)\
      .filter(Section.doc_id.in_(to_delete_docs))\
      .delete(synchronize_session=False)

    db.query(Document)\
      .filter(Document.id.in_(to_delete_docs))\
      .delete(synchronize_session=False)

    db.commit()
    db.close()
    print("✅  Deletion complete.")

if __name__ == "__main__":
    main()