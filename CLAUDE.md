# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all dependencies (including dev group)
uv sync --group dev

# Run the app (dev)
uv run streamlit run src/polychat/app.py

# Run the app via the installed CLI entrypoint
polychat                        # equivalent to the above
polychat --server.port=8080     # extra args forwarded to Streamlit

# Full gate (same as CI and pre-commit)
uv run pre-commit run --all-files

# Individual checks
uv run ruff check .
uv run ruff format --check .
uv run pyright

# Tests
uv run pytest                    # unit suite only (no network)
uv run pytest -m integration     # opt-in: requires API keys / Ollama
uv run pytest tests/unit/test_chain.py  # single test file

# Wire up git hooks after cloning
uv run pre-commit install
```

Every commit runs ruff → pyright → unit tests → hygiene checks. CI does the same on every push/PR.

## Architecture

This is a Streamlit RAG chatbot. The core principle: **`app.py` is UI wiring only — all retrieval logic lives in `rag/`**.

```
Streamlit UI (app.py)
    │
    ├── loaders/ ──► splitter.py ──► embeddings.py ──► vector_store.py
    │                                                        │
    └── i18n/                                          chain.py ──► llm.py
```

**Data flow for a user query:**
1. Source (PDF/CSV/TXT/URL/YouTube) → loader → `list[Document]`
2. Documents → `splitter.py` → chunked `list[Document]`
3. Chunks + embeddings → `VectorStoreFactory.build()` → vector index
4. At query time: `build_rag_chain()` creates a `RunnableWithMessageHistory` that rewrites follow-up questions using `condense_question_prompt`, retrieves chunks, then passes context + history to the LLM via `qa_prompt`.

**Key design decisions:**
- All factories (`get_llm`, `get_embeddings`) use lazy imports — provider packages are only imported if that provider is selected. This avoids import errors for providers not installed.
- `open_chroma_index` writes a `.fingerprint` sidecar alongside any persisted index to detect embedding-model mismatches across sessions.
- The persisted Chroma handle is cached per-process via `@st.cache_resource` in `app.py`, keyed on `(persist_dir, fingerprint, provider)`. This ensures one open SQLite connection regardless of how many times "Initialize RAG" is clicked.
- The i18n system (`i18n/__init__.py`) resolves the active locale from `streamlit.session_state` when Streamlit is running, or from a module-level variable otherwise — making all i18n helpers testable without a Streamlit runtime.
- `MissingAPIKeyError` is the single error type for missing API keys; `app.py` catches it and surfaces a localized message.
- `rag/prefs.py` persists `llm_provider`, `llm_model`, `embeddings_provider`, and `temperature` to `./polychat_prefs.json` (zero Streamlit imports — pure stdlib). `_bootstrap_state()` loads prefs on startup; `_save_prefs_callback()` writes them on every widget change. In Docker the file resolves to `/data/polychat_prefs.json` inside the volume.

**Adding a new LLM provider:** add to `LLMProvider` Literal in `llm.py`, extend `DEFAULT_MODELS`, add the branch in `get_llm()`.

**Chroma is the only vector store backend.** Ephemeral mode (`persist_index=False`) keeps the index in memory; persistent mode writes to `./chroma_db/`. The public API lives in `rag/vector_store.py`: `open_chroma_index`, `has_persisted_index`, `read_fingerprint`, `reset_chroma_index`.

**Adding a new language:** drop a `xx.json` next to `i18n/en.json` with a `_meta.name` field — no code changes.

## Docker

Two compose configurations are provided:

```bash
# Cloud providers (OpenAI / Anthropic / Groq)
cp .env.example .env          # fill in API keys
docker compose up             # build + run; open http://localhost:8501

# Fully-offline stack with Ollama
docker compose -f docker-compose.ollama.yml up
docker compose -f docker-compose.ollama.yml exec ollama ollama pull llama3.1:8b
# In the sidebar: Provider = Ollama · Model = llama3.1:8b · Embeddings = HuggingFace (local)
```

The image uses a multi-stage build (`uv` builder → `python:3.12-slim` runtime). Persisted indexes and HF model cache are stored under `/data` — mount a volume there in production.

The Docker image is published to GHCR automatically by `.github/workflows/docker-publish.yml` when a version tag is pushed (same trigger as the GitHub release).

## Branching model

- **`dev`** — default branch; all development work goes here.
- **`main`** — stable branch; only receives merges from `dev`.

Never commit directly to `main`. To release: bump the version in `pyproject.toml` on `dev`, commit and push to `dev`, then merge `dev` into `main` directly:

```bash
git checkout main
git merge dev --ff-only
git push origin main
git checkout dev
```

Pushing to `main` triggers `.github/workflows/release.yml`, which creates the Git tag, GitHub release, and builds and pushes the multi-platform Docker image to GHCR — all in one workflow.

## Testing conventions

- Unit tests live in `tests/unit/` and must not touch the network.
- `conftest.py` provides three key shared fixtures: `fake_embeddings` (a `DeterministicEmbeddings` that hashes text into floats — no model download), `tiny_locale_dir` (an isolated tmp locale dir with monkeypatching), and `FakeYouTubeAPI` / `FakeTranscriptList` for YouTube loader tests.
- Integration tests go in `tests/integration/` and are gated by `-m integration`.
- `pyright` is set to `strict` mode for `src/` and relaxed for `tests/`.
- `ruff` line length is 100; rule sets include `ANN` (annotations required everywhere except tests).
