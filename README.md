# PDF Q&A MCP Server

A locally-runnable [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that lets any MCP-compatible AI agent (Claude Desktop, etc.) ask natural language questions over a set of PDF documents and receive grounded answers with source attribution.

---

## Table of Contents

1. [Setup Instructions](#1-setup-instructions)
2. [Architecture Overview](#2-architecture-overview)
3. [Tool Documentation](#3-tool-documentation)
4. [Claude Desktop Integration](#4-claude-desktop-integration)
5. [Terminal Usage (query.py)](#5-terminal-usage-querypy)
6. [Example Interaction Log](#6-example-interaction-log)
7. [Vibe Coding Setup](#7-vibe-coding-setup)

---

## 1. Setup Instructions

### Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys) (used for both embeddings and answer generation)

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd Software-Engineer-AI-Take-Home-Aashay-Naik

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY

# 5. Add your PDF files
# Place all PDF documents in the pdfs/ directory
cp /path/to/your/documents/*.pdf pdfs/

# 6. Run the server
python -m src.server
```

On first run, the server automatically ingests all PDFs in `pdfs/` and prints progress to stderr. The ChromaDB vector store is persisted in `chroma_db/` so subsequent starts skip re-ingestion.

To force re-ingestion (e.g. after adding new PDFs):

```bash
python -m src.server --reingest
```

---

## 2. Architecture Overview

```
                              ┌─────────────────────────────────────────────────────────────────┐
                              │                        MCP Client                               │
                              │              (Claude Desktop / any MCP agent)                   │
                              └──────────────────────────┬──────────────────────────────────────┘
                                                         │  MCP stdio transport
                                                         ▼
                              ┌─────────────────────────────────────────────────────────────────┐
                              │                    src/server.py  (FastMCP)                     │
                              │   ┌──────────────────────┐   ┌──────────────────────────────┐   │
                              │   │  query_documents()   │   │     list_documents()         │   │
                              │   └──────────┬───────────┘   └──────────────────────────────┘   │
                              └──────────────┼──────────────────────────────────────────────────┘
                                             │
                                             ▼
                              ┌─────────────────────────────────────────────────────────────────┐
                              │                      src/qa.py  (RAG engine)                    │
                              │                                                                 │
                              │  question → OpenAI embed → ChromaDB query → top-k chunks        │
                              │           → RAG prompt → GPT-4o → answer + sources              │
                              └────────────────────────┬────────────────────────────────────────┘
                                                       │
                                        ┌──────────────┴──────────────┐
                                        ▼                             ▼
                              ┌─────────────────┐          ┌──────────────────────┐
                              │  chroma_db/     │          │  src/ingestion.py    │
                              │  (vector store) │◄─────────│  PDF → chunks →      │
                              └─────────────────┘          │  embeddings → upsert │
                                                           └──────────────────────┘
                                                                      ▲
                                                                      │
                                                           ┌──────────────────┐
                                                           │    pdfs/*.pdf    │
                                                           └──────────────────┘
```

### Component Details

| Component | File | Responsibility |
|---|---|---|
| MCP Server | `src/server.py` | FastMCP server; exposes tools; triggers ingestion on startup |
| Ingestion | `src/ingestion.py` | PyMuPDF → token-aware chunking → OpenAI embeddings → ChromaDB |
| Q&A Engine | `src/qa.py` | Embed question → retrieve chunks → RAG → GPT-4o answer |
| Vector Store | `chroma_db/` | Persisted ChromaDB with cosine-similarity index |

### Ingestion Pipeline

1. **Parse** — Each PDF is opened with PyMuPDF; text is extracted page-by-page.
2. **Chunk** — Pages are split into 400-token chunks with 50-token overlap using `tiktoken` (`cl100k_base` encoding).
3. **Embed** — Chunks are sent to OpenAI `text-embedding-3-small` in batches of 100.
4. **Store** — Embeddings + text + metadata (`source`, `page`, `chunk_index`) are upserted into a ChromaDB collection with cosine similarity.

### Query Flow

1. The question is embedded using the same `text-embedding-3-small` model.
2. ChromaDB returns the top-5 most semantically similar chunks (cross-document).
3. Each chunk is prefixed with its source label (`[Source: filename.pdf, Page N]`).
4. The assembled context + question is sent to `gpt-4o`.
5. GPT-4o generates an answer that cites specific pages. A `**References:**` block listing each document and page number is appended to the answer text, ensuring source attribution is always visible regardless of how the MCP client renders the response.

---

## 3. Tool Documentation

### `query_documents`

Ask a natural language question and receive a grounded answer from the indexed PDFs.

**Input schema:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `question` | string | yes | — | The natural language question |
| `num_sources` | integer | no | 5 | Number of source chunks to retrieve |
| `document_filter` | list[string] | no | null | Restrict search to specific PDF filenames (e.g. `["C18-1117.pdf"]`). Use `list_documents` to see available filenames. When omitted, all documents are searched. |

> **Note on `document_filter`:** Vector search works by semantic similarity of content, not by filename. If you ask "summarise C18-1117.pdf", the embedding of that text won't match content inside that file. Use `document_filter` to scope the search to a specific document when asking filename-specific questions.

**Output schema:**

```json
{
  "answer": "The grounded answer with inline citations...\n\n**References:**\n  - filename.pdf (page 3)\n  - filename.pdf (page 7)",
  "sources": [
    {"document": "filename.pdf", "page": 3}
  ]
}
```

The `answer` field always includes a `**References:**` block at the end so that page numbers are visible in every MCP client, including Claude Desktop.

**Example calls:**

```json
{
  "name": "query_documents",
  "arguments": {
    "question": "What are the main risk factors described in the annual report?",
    "num_sources": 5
  }
}
```

```json
{
  "name": "query_documents",
  "arguments": {
    "question": "Provide a brief summary of this paper",
    "document_filter": ["C18-1117.pdf"]
  }
}
```

---

### `list_documents`

Returns the filenames of all PDFs currently indexed in the vector store.

**Input schema:** none

**Output schema:**

```json
["document_a.pdf", "document_b.pdf", "document_c.pdf"]
```

---

## 4. Claude Desktop Integration

Add the following to your Claude Desktop MCP config file:

**Mac:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "pdf-qa-server": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["/absolute/path/to/Software-Engineer-AI-Take-Home-Aashay-Naik/src/server.py"],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

Restart Claude Desktop. The server will appear as an available tool.

---

## 5. Terminal Usage (query.py)

`query.py` is a standalone CLI script that calls the Q&A engine directly — no MCP client required. It is useful for testing queries and generating example output without needing Claude Desktop.

> The MCP server does **not** need to be running when using `query.py`. It bypasses the MCP protocol entirely and calls `src/qa.py` directly.

### Basic query

```bash
python query.py "Your question here"
```

### Interactive prompt

```bash
python query.py
# Prompts: Question:
```

### Scope query to a specific document

Use `--doc` when your question is about a particular file. Without this flag, semantic search may return chunks from unrelated documents since it matches by content similarity, not filename.

```bash
python query.py --doc C18-1117.pdf "Provide a brief summary"
```

### Scope query to multiple documents

```bash
python query.py --doc W18-4401.pdf --doc D19-1539.pdf "What do these papers have in common?"
```

---

## 6. Example Interaction Log

> The examples below show actual output generated using the author's PDF corpus (NLP research papers). Outputs will differ based on the PDFs you place in the `pdfs/` directory.

---

**Q1 — Single-document lookup (no filter)**

*Question:* `Explain the concept of semantic shifts`

*Response:*
```
--- Answer ---
The concept of semantic shifts, also known as semantic change, refers to the evolution of word meanings over time. Bloomfield (1933) defines it as “innovations which change the lexical meaning rather than the grammatical function of a form” [C18-1117.pdf, Page 2]. Semantic shifts occur due to a variety of factors, including linguistic, psychological, and sociocultural influences [C18-1117.pdf, Page 2]. These shifts can lead to changes in the meaning of a word, such as becoming more specific or more general, known as narrowing or broadening. An example of this would be the Old English word "mete" meaning 'food', which later became the English word "meat" specifically referring to 'edible flesh' [C18-1117.pdf, Page 2]. 

Semantic shifts can also be linked with changes in the speaker’s attitudes or changes in the environment, leading to shifts often classified into linguistic drifts and cultural shifts. Linguistic drifts are slow and regular changes in the core meaning of words, whereas cultural shifts are changes in the associations of a word as determined by cultural factors [C18-1117.pdf, Page 2]. Modern studies use large corpora and distributional semantics approaches to monitor these shifts through changes in word usage patterns over time [C18-1117.pdf, Page 3].

**References:**
  - C18-1117.pdf (page 2)
  - C18-1117.pdf (page 3)
  - C18-1117.pdf (page 10)
```

---

**Q2 — Specific document lookup with filter**

*Question:* `Provide a brief summary` *(with `document_filter: ["C18-1117.pdf"]`)*

*CLI:* `python query.py --doc C18-1117.pdf "Provide a brief summary"`

*Response:*
```
The document is a survey of research in the emerging field of computational detection of semantic shifts using diachronic word embeddings. It discusses the nature of semantic changes over time, the data sources used to study these shifts, and the various distributional approaches to model these changes, from traditional frequentist methods to modern prediction-based models. Despite recent advancements and academic exchanges, the field remains somewhat fragmented, partly due to a lack of dedicated workshops and shared tasks that focus on word meaning changes over time. The document also mentions the development of the field through influential events and suggests that community-driven efforts, like organizing focused workshops, could promote further research. The survey highlights the importance of identifying groups of words that shift together and the lack of algorithms that are sense-aware, pointing towards potential avenues for future research (Source: C18-1117.pdf, Pages 10-11).

**References:**
  - C18-1117.pdf (page 10)
  - C18-1117.pdf (page 14)
  - C18-1117.pdf (page 11)
```

---

**Q3 — Cross-document lookup**

*Question:* `What is the difference between the retrieve and refine model and the two tower model?`

*Response:*
```
The Retrieve and Refine model (RetNRef) and the two tower model are two different approaches applied in distinct tasks. 

1. **Retrieve and Refine model (RetNRef):**
   - RetNRef is designed for dialogue systems, specifically to refine dialogue generation. It uses a retrieval mechanism to fetch candidate responses from a training set. Then, these candidates are reranked for similarity to the target label and used to train a Seq2Seq model with appended chosen utterances ([Source: W18-5713.pdf, Page 2]).
   - It includes variants like RetNRef+ and RetNRef++ which focus more on the retrieval process and fixing retrieval copy errors, respectively ([Source: W18-5713.pdf, Page 2]).

2. **Two tower model:**
   - This model is built for predicting tokens in a sentence, leveraging a dual self-attentional architecture operating in forward and backward directions. It's structured to compute probability distributions over token sequences and is used for pretraining on language tasks ([Source: D19-1539.pdf, Page 2]).
   - The model combines two sets of representations without containing information about the current target token, making it suitable for cloze tasks ([Source: D19-1539.pdf, Page 2]).

In summary, RetNRef is specifically crafted for dialogue system applications with emphasis on retrieval-based refinement, whereas the two tower model is focused on token prediction and uses a bidirectional self-attention mechanism, suitable for tasks like cloze tests.

**References:**
  - W18-5713.pdf (page 2)
  - D19-1539.pdf (page 2)
  - D19-1539.pdf (page 4)
  - W18-5713.pdf (page 3)
```

---

**Q4 — List all available documents**

*Question:* `List the documents available to query from`

*Response:*
```
The following 5 document(s) are available:

  - C18-1117.pdf
  - D19-1539.pdf
  - P19-1164.pdf
  - W18-4401.pdf
  - W18-5713.pdf
```

---

## 6. Vibe Coding Setup

### Tools Used

- **Claude Code** (primary) — used for planning, answering implementation questions, and generating code scaffolding
- **Claude Desktop** — used for end-to-end testing of the MCP server once it was running

### How I Used the AI

I started by sharing the assignment brief and the document for reference and asking Claude Code to help me think through the architecture before writing any code. I had a rough idea of what I wanted — a RAG pipeline on top of PDFs exposed as an MCP tool — but wasn't sure about some of the specifics.

Some things I asked about during the planning phase:

- **Which MCP framework to use** — I asked whether to use the raw `mcp` SDK or FastMCP. The AI explained that FastMCP wraps the official SDK with a cleaner decorator-based API, which made it the obvious choice for keeping the server code readable.
- **How chunking works** — I didn't fully understand why overlap matters in text chunking, so I asked the AI to explain it. The short answer: without overlap, a sentence that falls across a chunk boundary gets cut off in both chunks and may never be fully retrieved.
- **Why ChromaDB needs cosine similarity set explicitly** — I noticed the metadata flag in the generated code and asked what happens without it. The AI explained that ChromaDB defaults to L2 (Euclidean) distance, which doesn't work well with normalised embedding vectors — cosine is the right choice here.
- **What stdio transport means for MCP** — I wasn't familiar with how stdio-based servers work. The AI clarified that the MCP protocol communicates over stdin/stdout, which means any `print()` to stdout would corrupt the protocol stream — all logs had to go to stderr instead.

### What I Decided Myself

- **Skipping re-ingestion on subsequent starts** — I decided the server should check whether the vector store is already populated and skip ingestion if so, rather than rebuilding the index every time it starts. For a corpus of 4–5 PDFs this matters less, but it felt like the right default — ingestion is the slow, expensive part and should only run when needed. The `--reingest` flag gives an explicit escape hatch.

- **Separating ingestion, Q&A, and the server into three modules** — I could have put everything in one file, but I deliberately split it so that `ingestion.py` and `qa.py` can be imported and tested independently without starting the MCP server. This made debugging much easier — I could run `query.py` directly against the Q&A engine without any MCP overhead.

- **Added the `document_filter` parameter** — After running my first query (`python query.py "Provide brief summary of C18-1117.pdf"`), the answer came back from completely different documents. I realised semantic search matches by content similarity, not filename, so asking about a file by name doesn't scope the search to it. I added the `--doc` flag and `document_filter` parameter to restrict the ChromaDB query to specific documents when needed.

### Overall View on AI Tooling in Software Engineering

Honestly, having an AI assistant made this project a lot more approachable than it would have been otherwise. There were parts of this — like the MCP protocol, ChromaDB internals, or how embeddings and vector search fit together — where I would have spent hours reading documentation. Being able to just ask questions and get clear, specific answers saved a lot of time.

That said, I found that the AI is most useful when you stay in the loop rather than just accepting whatever it generates. A few times the generated code looked right but had subtle issues I only caught by actually running it and seeing what broke. The debugging and decision-making still felt like mine — the AI helped me move faster, but I had to understand what was happening well enough to know when something was wrong. I think that's probably the right way to use these tools: let them handle the parts you already understand well enough to verify, and make sure you're still the one thinking through the harder decisions.
