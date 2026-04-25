"""
Q&A engine: embed question → retrieve chunks → RAG prompt → OpenAI answer.
"""

from openai import OpenAI

from src.ingestion import _get_client, _get_collection

MODEL = "gpt-4o"
MAX_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a precise document assistant. Answer questions using only the "
    "provided document excerpts. Cite the source document and page number for "
    "every claim you make. If the excerpts do not contain enough information to "
    "answer the question fully, state that clearly rather than speculating."
)


def query_documents(
    question: str,
    db_path: str,
    openai_api_key: str,
    num_sources: int = 5,
    document_filter: list[str] | None = None,
) -> dict:
    """
    Retrieve the most relevant chunks for *question* and generate a grounded
    answer using OpenAI GPT-4o.

    Args:
        document_filter: Optional list of PDF filenames to restrict the search to.
                         When provided, only chunks from those documents are considered.

    Returns:
        {
            "answer": str,
            "sources": [{"document": str, "page": int}, ...]
        }
    """
    openai_client = OpenAI(api_key=openai_api_key)

    # 1. Embed the question
    q_embedding = (
        openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[question],
        )
        .data[0]
        .embedding
    )

    # 2. Retrieve top-k chunks from ChromaDB
    collection = _get_collection(_get_client(db_path))
    total_docs = collection.count()
    if total_docs == 0:
        return {
            "answer": "No documents have been indexed yet. Please run ingestion first.",
            "sources": [],
        }

    where = (
        {"source": {"$in": document_filter}}
        if document_filter
        else None
    )

    k = min(num_sources, total_docs)
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=k,
        include=["documents", "metadatas"],
        where=where,
    )

    chunks: list[str] = results["documents"][0]
    metadatas: list[dict] = results["metadatas"][0]

    # 3. Build context with inline source labels
    context_parts = []
    seen_sources: list[dict] = []
    for chunk, meta in zip(chunks, metadatas):
        label = f"[Source: {meta['source']}, Page {meta['page']}]"
        context_parts.append(f"{label}\n{chunk}")
        source = {"document": meta["source"], "page": meta["page"]}
        if source not in seen_sources:
            seen_sources.append(source)

    context = "\n\n---\n\n".join(context_parts)

    # 4. Call GPT-4o
    response = openai_client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Document excerpts:\n\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
    )

    answer_text = response.choices[0].message.content

    # Append a formatted references block so page numbers are always visible
    # regardless of how the MCP client renders the response.
    references = "\n".join(
        f"  - {s['document']} (page {s['page']})" for s in seen_sources
    )
    answer_with_sources = f"{answer_text}\n\n**References:**\n{references}"

    return {
        "answer": answer_with_sources,
        "sources": seen_sources,
    }
