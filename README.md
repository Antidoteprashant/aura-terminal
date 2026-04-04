# RAG Terminal

A terminal-themed Retrieval-Augmented Generation (RAG) web app. Upload study documents (PDF, TXT, Markdown), ask questions in natural language, and get cited answers streamed in real time — all running locally with zero API cost, or deployable to the cloud.

```
╔══════════════════════════════════════════════════╗
║          AURA  DOCUMENT Q&A TERMINAL             ║
║     Powered by Gemma 4 · ChromaDB · RAG          ║
╚══════════════════════════════════════════════════╝
```

---

## Features

- **Terminal UI** — dark CRT aesthetic, scanlines, typewriter streaming, blinking cursor
- **RAG pipeline** — parse → chunk → embed → vector search → generate
- **Hybrid deployment** — Ollama + ChromaDB locally; Groq + Pinecone on the cloud
- **Automatic LLM fallback** — if Ollama is offline, routes to Groq automatically
- **Conversation memory** — follow-up questions work (last 5 turns kept in context)
- **Drag-and-drop upload** — or use the `/upload` command
- **SSE streaming** — answers stream token-by-token, no page reloads
- **No frameworks** — vanilla HTML/CSS/JS frontend, no npm, no build step

---

## Stack

| Layer | Local | Hosted |
|---|---|---|
| LLM | Ollama (Gemma 4 / any model) | Groq via `langchain-groq` |
| Vector store | ChromaDB (embedded) | Pinecone (serverless) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | same |
| Backend | Flask 3 + SSE | same |
| Frontend | Vanilla HTML/CSS/JS | same |

---

## Quick Start (Local)

### 1. Install Ollama and pull a model

```bash
# macOS
brew install ollama
ollama serve

# Pull a model (Gemma 4 recommended)
ollama pull gemma4:e4b
# or any smaller model
ollama pull gemma3:4b
```

### 2. Clone and set up the project

```bash
git clone https://github.com/Antidoteprashant/rag-terminal
cd rag-terminal

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — set OLLAMA_MODEL to whatever you pulled
```

### 4. Run

```bash
python run.py
# Open http://localhost:5001
```

---

## Terminal Commands

| Command | What it does |
|---|---|
| `<any text>` | Ask a question about your uploaded documents |
| `/upload` | Open file picker to upload a PDF, TXT, or MD file |
| `/docs` or `/ls` | List all uploaded documents |
| `/summarize <filename>` | Summarize a specific document |
| `/delete <filename>` | Remove a document and its embeddings |
| `/status` | Show LLM backend, model, doc count |
| `/clear` | Clear the terminal output |
| `/help` | Show all available commands |

You can also drag and drop files anywhere on the page to upload them.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the values you need.

### Local mode (default)

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:e4b

# Optional: add a Groq key as an automatic fallback if Ollama goes offline
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant
```

### Hosted mode (Render / any cloud)

Set `RENDER=1` or `PRODUCTION=1` in your environment. The app will refuse to start without all three hosted keys.

```env
RENDER=1

GROQ_API_KEY=gsk_...          # https://console.groq.com
GROQ_MODEL=llama-3.1-8b-instant

PINECONE_API_KEY=pcsk_...     # https://app.pinecone.io
PINECONE_INDEX=rag-terminal   # must exist in your Pinecone project
```

### All variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `gemma4-e2b` | Ollama model name |
| `GROQ_API_KEY` | *(empty)* | Groq API key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model ID |
| `PINECONE_API_KEY` | *(empty)* | Pinecone API key |
| `PINECONE_INDEX` | `rag-terminal` | Pinecone index name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |
| `CHROMA_COLLECTION_NAME` | `documents` | ChromaDB collection |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `FLASK_HOST` | `0.0.0.0` | Flask bind address |
| `FLASK_PORT` | `5000` | Flask port |
| `FLASK_DEBUG` | `true` | Debug mode (auto-off in hosted) |
| `UPLOAD_FOLDER` | `./uploads` | Temp upload directory |
| `MAX_CONTENT_LENGTH` | `16777216` | Max file size (16 MB) |
| `ALLOWED_EXTENSIONS` | `pdf,txt,md` | Accepted file types |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload and ingest a document |
| `POST` | `/api/query` | Ask a question (SSE stream) |
| `POST` | `/api/summarize` | Summarize a document (SSE stream) |
| `GET` | `/api/documents` | List all uploaded documents |
| `DELETE` | `/api/documents/<doc_id>` | Delete a document |
| `GET` | `/api/health` | LLM/backend status |
| `DELETE` | `/api/conversations` | Clear conversation history |

### SSE stream format

Both `/api/query` and `/api/summarize` stream [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events):

```
data: {"token": "Hello"}
data: {"token": " world"}
data: {"done": true, "sources": [{"doc_id": "...", "filename": "notes.pdf"}]}
```

---

## Project Structure

```
rag-terminal/
├── run.py                      # Entry point — env detection + app start
├── requirements.txt
├── .env.example
├── app/
│   ├── __init__.py             # Flask app factory
│   ├── routes.py               # API endpoints
│   ├── config.py               # Config from .env
│   ├── ingestion/
│   │   ├── parser.py           # PDF / TXT / MD → plain text
│   │   ├── chunker.py          # Text → overlapping chunks
│   │   ├── embedder.py         # Chunks → 384-dim vectors (singleton)
│   │   ├── vector_store.py     # ChromaDB wrapper (local)
│   │   ├── pinecone_store.py   # Pinecone wrapper (hosted)
│   │   └── store.py            # Proxy — routes to active backend
│   ├── rag/
│   │   ├── ollama_client.py    # Ollama streaming client
│   │   ├── groq_client.py      # Groq streaming client (plain requests)
│   │   ├── llm.py              # LLM router (Ollama → Groq → langchain_groq)
│   │   ├── prompts.py          # QA + summarize prompt templates
│   │   └── rag_chain.py        # Full RAG pipeline
│   ├── models/
│   │   └── conversation.py     # In-memory conversation history
│   └── static/
│       ├── index.html
│       ├── terminal.css
│       └── app.js
├── tests/
│   ├── test_parser.py
│   ├── test_chunker.py
│   └── test_rag.py
├── uploads/                    # Temp files (gitignored)
└── chroma_db/                  # ChromaDB data (gitignored)
```

---

## Deploying to Render

`render.yaml` is included — Render will auto-configure the service from it.

### Steps

1. Create a free [Pinecone](https://app.pinecone.io) index named `rag-terminal` with **dimension 384** and **metric cosine** before deploying.
2. Push the repo to GitHub.
3. Go to [Render](https://render.com) → **New** → **Blueprint** → connect the repo.
   Render reads `render.yaml` and creates the service automatically.
4. In the Render dashboard, open the service → **Environment** and set the two secret keys (marked `sync: false` in `render.yaml`):
   - `GROQ_API_KEY` — from [console.groq.com](https://console.groq.com)
   - `PINECONE_API_KEY` — from [app.pinecone.io](https://app.pinecone.io)
5. Trigger a deploy. The app will start on Groq + Pinecone automatically.

If any required key is missing the process exits immediately with a message listing exactly which variables to add.

### Notes

- **Free tier** spins down after 15 minutes of inactivity (cold start ~30 s).
- Uploaded files land in `/tmp/uploads` — they are deleted after processing, so this is fine.
- ChromaDB is **not** used on Render; all vectors go to Pinecone.
- To upgrade to always-on, change `plan: free` → `plan: starter` in `render.yaml`.

---

## Performance (local, CPU-only, 8 GB RAM)

| Operation | Expected time |
|---|---|
| File upload + chunking | 1–3 s |
| Embedding generation | 2–5 s |
| Vector search (ChromaDB) | < 100 ms |
| LLM response (Gemma 4, quantized, CPU) | 15–30 s |

LLM inference is the bottleneck. The typewriter animation makes the wait feel shorter because tokens stream in as they are generated.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused` on port 11434 | Run `ollama serve` |
| Model not found | Run `ollama pull <model>` |
| Out of memory | Use a smaller model: `ollama pull gemma3:2b` |
| ChromaDB lock error | Delete `./chroma_db` and re-upload |
| Port already in use | Set `FLASK_PORT` in `.env` |
| Hosted startup error | Check that all three hosted env vars are set |
| Pinecone 404 | Make sure the index name in `PINECONE_INDEX` exists |

---

## License

MIT
