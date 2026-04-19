# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all dependencies (including dev group)
uv sync --group dev

# Run the app
uv run streamlit run src/polychat/app.py

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
- All factories (`get_llm`, `get_embeddings`, `VectorStoreFactory`) use lazy imports — provider packages are only imported if that provider is selected. This avoids import errors for providers not installed.
- `VectorStoreFactory` writes a `.fingerprint` sidecar alongside any persisted index to detect embedding-model mismatches across sessions.
- The i18n system (`i18n/__init__.py`) resolves the active locale from `streamlit.session_state` when Streamlit is running, or from a module-level variable otherwise — making all i18n helpers testable without a Streamlit runtime.
- `MissingAPIKeyError` is the single error type for missing API keys; `app.py` catches it and surfaces a localized message.

**Adding a new LLM provider:** add to `LLMProvider` Literal in `llm.py`, extend `DEFAULT_MODELS`, add the branch in `get_llm()`.

**Adding a new vector store backend:** extend `VectorStoreBackend` Literal in `vector_store.py`, add `_build_X` / `_load_X` methods, wire into `build()` and `load()`.

**Adding a new language:** drop a `xx.json` next to `i18n/en.json` with a `_meta.name` field — no code changes.

## Branching model

- **`dev`** — default branch; all development work goes here.
- **`main`** — stable branch; only receives PRs from `dev`.

Never commit directly to `main`. To release: bump the version in `pyproject.toml` on `dev`, then open a PR → `main`. Merging that PR triggers `.github/workflows/release.yml`, which creates the Git tag and GitHub release automatically.

## Testing conventions

- Unit tests live in `tests/unit/` and must not touch the network.
- `conftest.py` provides three key shared fixtures: `fake_embeddings` (a `DeterministicEmbeddings` that hashes text into floats — no model download), `tiny_locale_dir` (an isolated tmp locale dir with monkeypatching), and `FakeYouTubeAPI` / `FakeTranscriptList` for YouTube loader tests.
- Integration tests go in `tests/integration/` and are gated by `-m integration`.
- `pyright` is set to `strict` mode for `src/` and relaxed for `tests/`.
- `ruff` line length is 100; rule sets include `ANN` (annotations required everywhere except tests).
