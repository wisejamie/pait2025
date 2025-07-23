import os, sys
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
from app.db.models import *
from app.db.session import SessionLocal
import sys
import pprint

def main():
    db = SessionLocal()
    pprint.pprint([{"id": d.id, "title": d.title} for d in db.query(Document).all()])
    db.close()

if __name__ == "__main__":
    main()