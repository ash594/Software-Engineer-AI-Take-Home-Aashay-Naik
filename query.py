"""
Quick CLI to query the indexed PDFs directly.

Usage:
    python query.py "Your question here"
    python query.py --doc C18-1117.pdf "Summarise this document"
    python query.py --list                 # show all indexed documents
    python query.py                        # interactive prompt
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import os
sys.path.insert(0, str(Path(__file__).parent))

from src.qa import query_documents
from src.ingestion import is_collection_populated, get_indexed_documents

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "chroma_db")

if not is_collection_populated(CHROMA_DB_PATH):
    print("No documents indexed yet. Run: python -m src.server --reingest")
    sys.exit(1)

# Parse flags
args = sys.argv[1:]

# --list: print indexed documents and exit
if "--list" in args:
    docs = get_indexed_documents(CHROMA_DB_PATH)
    print(f"\n{len(docs)} document(s) indexed:")
    for d in docs:
        print(f"  - {d}")
    sys.exit(0)

# Parse optional --doc flag(s)
document_filter = None
filtered_args = []
i = 0
while i < len(args):
    if args[i] == "--doc" and i + 1 < len(args):
        document_filter = document_filter or []
        document_filter.append(args[i + 1])
        i += 2
    else:
        filtered_args.append(args[i])
        i += 1

question = " ".join(filtered_args) if filtered_args else input("Question: ")

# Detect listing intent and short-circuit to get_indexed_documents
_list_keywords = {"list", "show", "what", "which", "available", "documents", "files", "pdfs"}
_question_words = set(question.lower().split())
if len(_list_keywords & _question_words) >= 2 and not document_filter:
    docs = get_indexed_documents(CHROMA_DB_PATH)
    print("\n--- Answer ---")
    print(f"The following {len(docs)} document(s) are available:\n")
    for d in docs:
        print(f"  - {d}")
    sys.exit(0)

result = query_documents(
    question=question,
    db_path=CHROMA_DB_PATH,
    openai_api_key=OPENAI_API_KEY,
    document_filter=document_filter,
)

print("\n--- Answer ---")
print(result["answer"])
