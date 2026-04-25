"""
MCP server entry point.

Run:
    python -m src.server              # normal start (ingests only if DB is empty)
    python -m src.server --reingest   # force re-ingestion
"""

import os
import sys
from pathlib import Path

# Make project root importable regardless of working directory
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PDFS_DIR = os.environ.get("PDFS_DIR", str(PROJECT_ROOT / "pdfs"))
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", str(PROJECT_ROOT / "chroma_db"))

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill in your key.")

from src.ingestion import ingest_pdfs, is_collection_populated, get_indexed_documents
from src.qa import query_documents as _query_documents

# ---------------------------------------------------------------------------
# Startup ingestion
# ---------------------------------------------------------------------------

_reingest = "--reingest" in sys.argv

if _reingest or not is_collection_populated(CHROMA_DB_PATH):
    reason = "forced re-ingestion" if _reingest else "collection is empty"
    print(f"[pdf-qa-server] Ingesting PDFs ({reason})...", file=sys.stderr)
    total = ingest_pdfs(PDFS_DIR, CHROMA_DB_PATH, OPENAI_API_KEY)
    print(f"[pdf-qa-server] Done — {total} chunks indexed.", file=sys.stderr)
else:
    docs = get_indexed_documents(CHROMA_DB_PATH)
    print(
        f"[pdf-qa-server] Collection ready ({len(docs)} document(s)). "
        "Use --reingest to refresh.",
        file=sys.stderr,
    )

# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("pdf-qa-server")


@mcp.tool()
def query_documents(
    question: str,
    num_sources: int = 5,
    document_filter: list[str] | None = None,
) -> dict:
    """
    Ask a content question and receive a grounded answer derived from the indexed
    PDF documents. Use this for questions about what the documents say or contain.

    Do NOT use this to list available documents — call list_documents() instead.

    Args:
        question: The natural language question to answer.
        num_sources: Number of source chunks to retrieve for context (default 5).
        document_filter: Optional list of PDF filenames to restrict the search to
                         (e.g. ["C18-1117.pdf"]). Use list_documents() to see
                         available filenames. When omitted, all documents are searched.

    Returns:
        A dict with:
        - answer (str): The grounded answer attributed to source documents.
        - sources (list[dict]): Each entry has 'document' (filename) and
          'page' (page number) indicating where the answer was drawn from.
    """
    return _query_documents(
        question=question,
        db_path=CHROMA_DB_PATH,
        openai_api_key=OPENAI_API_KEY,
        num_sources=num_sources,
        document_filter=document_filter,
    )


@mcp.tool()
def list_documents() -> list:
    """
    Return the filenames of ALL PDF documents currently indexed in the vector
    store. Use this whenever the user asks what documents are available, what
    files are indexed, or wants to know what they can query. This always returns
    the complete list — do not use query_documents() for listing purposes.

    Returns:
        A list of PDF filenames (e.g. ["report.pdf", "whitepaper.pdf"]).
    """
    return get_indexed_documents(CHROMA_DB_PATH)


if __name__ == "__main__":
    mcp.run()
