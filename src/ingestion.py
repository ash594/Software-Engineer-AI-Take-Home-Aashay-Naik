"""
PDF ingestion pipeline: parse → chunk → embed → store in ChromaDB.
"""

import os
import hashlib
import fitz  # PyMuPDF
import tiktoken
import chromadb
from openai import OpenAI

COLLECTION_NAME = "pdf_docs"
CHUNK_SIZE = 400       # tokens per chunk
CHUNK_OVERLAP = 50     # overlap between consecutive chunks
EMBED_BATCH_SIZE = 100 # chunks per OpenAI embedding API call


def _get_client(db_path: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=db_path)


def _get_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _chunk_text(
    text: str,
    encoding: tiktoken.Encoding,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into token-bounded chunks with overlap."""
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(encoding.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def _embed_batch(texts: list[str], client: OpenAI) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


def ingest_pdfs(pdfs_dir: str, db_path: str, openai_api_key: str) -> int:
    """
    Parse all PDFs in pdfs_dir, chunk, embed, and upsert into ChromaDB.

    Returns the total number of chunks indexed.
    """
    openai_client = OpenAI(api_key=openai_api_key)
    chroma_client = _get_client(db_path)
    collection = _get_collection(chroma_client)
    encoding = tiktoken.get_encoding("cl100k_base")

    pdf_files = sorted(f for f in os.listdir(pdfs_dir) if f.lower().endswith(".pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in '{pdfs_dir}'")

    total_chunks = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdfs_dir, pdf_file)
        doc = fitz.open(pdf_path)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text("text").strip()
            if not page_text:
                continue

            for chunk_idx, chunk in enumerate(_chunk_text(page_text, encoding)):
                if not chunk.strip():
                    continue
                chunk_id = hashlib.md5(
                    f"{pdf_file}:p{page_num + 1}:c{chunk_idx}".encode()
                ).hexdigest()
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append(
                    {
                        "source": pdf_file,
                        "page": page_num + 1,
                        "chunk_index": chunk_idx,
                    }
                )

        doc.close()

        if not documents:
            print(f"  Skipping {pdf_file}: no extractable text")
            continue

        # Embed in batches to stay within API limits
        embeddings: list[list[float]] = []
        for i in range(0, len(documents), EMBED_BATCH_SIZE):
            embeddings.extend(_embed_batch(documents[i : i + EMBED_BATCH_SIZE], openai_client))

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total_chunks += len(documents)
        print(f"  {pdf_file}: {len(documents)} chunks indexed")

    return total_chunks


def is_collection_populated(db_path: str) -> bool:
    """Return True if the ChromaDB collection already has documents."""
    try:
        client = _get_client(db_path)
        return _get_collection(client).count() > 0
    except Exception:
        return False


def get_indexed_documents(db_path: str) -> list[str]:
    """Return sorted list of unique PDF filenames in the collection."""
    client = _get_client(db_path)
    collection = _get_collection(client)
    results = collection.get(include=["metadatas"])
    return sorted({m["source"] for m in results["metadatas"]})
