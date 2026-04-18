"""Streamlit entrypoint for PolyChat.

Run with::

    streamlit run src/polychat/app.py

This module stays thin on purpose: widgets, state wiring, and error routing
only. All RAG logic lives in :mod:`polychat.rag`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.vectorstores import VectorStore

from .i18n import DEFAULT_LOCALE, available_locales, set_locale, t
from .rag.chain import DEFAULT_HISTORY_KEY, build_rag_chain
from .rag.embeddings import (
    EmbeddingsProvider,
    MissingAPIKeyError,
    fingerprint,
    get_embeddings,
)
from .rag.llm import LLMProvider, available_models, get_llm
from .rag.loaders import SourceKind, load_source
from .rag.loaders.youtube import (
    AgeRestrictedError,
    InvalidYouTubeURLError,
    RequestBlockedError,
    TranscriptUnavailableError,
    VideoUnavailableError,
    YouTubeLoaderError,
)
from .rag.splitter import split_documents
from .rag.vector_store import (
    EmbeddingsMismatchError,
    VectorStoreBackend,
    VectorStoreFactory,
)

logger = logging.getLogger(__name__)

load_dotenv()

SESSION_ID = "default"
SOURCE_KINDS: list[SourceKind] = ["pdf", "csv", "txt", "site", "youtube"]
FILE_KINDS = {"pdf", "csv", "txt"}
FILE_EXT: dict[str, list[str]] = {"pdf": ["pdf"], "csv": ["csv"], "txt": ["txt", "md"]}
EMBEDDINGS_PROVIDERS: list[EmbeddingsProvider] = ["openai", "huggingface_local"]
LLM_PROVIDERS: list[LLMProvider] = ["openai", "anthropic", "groq", "ollama"]
VECTOR_BACKENDS: list[VectorStoreBackend] = ["chroma", "faiss", "qdrant"]


def main() -> None:
    st.set_page_config(
        page_title="PolyChat",
        page_icon=":robot_face:",
        layout="wide",
    )

    _bootstrap_state()
    _render_sidebar()
    _render_main()


# ----------------------------------------------------------------- state


def _bootstrap_state() -> None:
    if "locale" not in st.session_state:
        st.session_state["locale"] = DEFAULT_LOCALE
    set_locale(st.session_state["locale"])
    st.session_state.setdefault("rag_chain", None)
    st.session_state.setdefault("retriever_ready", False)
    st.session_state.setdefault("source_kind", "pdf")
    st.session_state.setdefault("llm_provider", "openai")
    st.session_state.setdefault("llm_model", available_models("openai")[0])
    st.session_state.setdefault("embeddings_provider", "huggingface_local")
    st.session_state.setdefault("vector_backend", "chroma")
    st.session_state.setdefault("persist_index", False)
    st.session_state.setdefault("temperature", 0.3)
    # Instantiate the history lazily; it's owned by session_state internally.
    StreamlitChatMessageHistory(key=DEFAULT_HISTORY_KEY)


# ----------------------------------------------------------------- sidebar


def _render_sidebar() -> None:
    with st.sidebar:
        tab_files, tab_models = st.tabs([t("sidebar.tab.files"), t("sidebar.tab.models")])
        with tab_files:
            _render_files_tab()
        with tab_models:
            _render_models_tab()


def _render_files_tab() -> None:
    kind: SourceKind = st.selectbox(
        t("sidebar.source_type"),
        options=SOURCE_KINDS,
        index=SOURCE_KINDS.index(st.session_state["source_kind"]),
        format_func=lambda v: t(f"sidebar.source.{v}"),
        key="source_kind",
    )

    uploaded: list[Any] = []
    url: str = ""
    if kind in FILE_KINDS:
        uploaded = (
            st.file_uploader(
                t("sidebar.upload.files"),
                type=FILE_EXT[kind],
                accept_multiple_files=True,
                key=f"uploader_{kind}",
            )
            or []
        )
    elif kind == "site":
        url = st.text_input(t("sidebar.upload.site"), key="url_site")
    elif kind == "youtube":
        url = st.text_input(t("sidebar.upload.youtube"), key="url_youtube")

    st.selectbox(
        t("sidebar.vector_store"),
        options=VECTOR_BACKENDS,
        index=VECTOR_BACKENDS.index(st.session_state["vector_backend"]),
        key="vector_backend",
    )
    st.checkbox(t("sidebar.persist"), key="persist_index")

    col_init, col_clear = st.columns(2)
    if col_init.button(t("sidebar.button.init"), type="primary", use_container_width=True):
        _on_init_rag(kind=kind, uploaded=uploaded, url=url)
    if col_clear.button(t("sidebar.button.clear"), use_container_width=True):
        _on_clear_history()


def _render_models_tab() -> None:
    locales = available_locales()
    codes = list(locales.keys())
    current = st.session_state.get("locale", DEFAULT_LOCALE)
    new_locale = st.selectbox(
        t("sidebar.language"),
        options=codes,
        index=codes.index(current) if current in codes else 0,
        format_func=lambda c: locales[c],
    )
    if new_locale != current:
        set_locale(new_locale)
        st.rerun()

    provider: LLMProvider = st.selectbox(
        t("sidebar.llm_provider"),
        options=LLM_PROVIDERS,
        index=LLM_PROVIDERS.index(st.session_state["llm_provider"]),
        format_func=lambda v: v.capitalize(),
        key="llm_provider",
    )
    models = available_models(provider)
    current_model = st.session_state.get("llm_model", models[0])
    model_index = models.index(current_model) if current_model in models else 0
    st.selectbox(t("sidebar.model"), options=models, index=model_index, key="llm_model")

    st.selectbox(
        t("sidebar.embeddings"),
        options=EMBEDDINGS_PROVIDERS,
        index=EMBEDDINGS_PROVIDERS.index(st.session_state["embeddings_provider"]),
        format_func=lambda v: t(f"sidebar.embeddings.{v}"),
        key="embeddings_provider",
    )
    st.slider(
        t("sidebar.temperature"),
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="temperature",
    )


# ------------------------------------------------------------- main / chat


def _render_main() -> None:
    st.title(t("app.title"))
    st.caption(t("app.subtitle"))

    history = StreamlitChatMessageHistory(key=DEFAULT_HISTORY_KEY)
    for message in history.messages:
        role = "user" if message.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(str(message.content))

    if not st.session_state.get("retriever_ready"):
        st.info(t("chat.not_ready"))
        return

    user_input = st.chat_input(t("chat.placeholder"))
    if not user_input:
        return

    with st.chat_message("user"):
        st.markdown(user_input)

    chain = st.session_state.get("rag_chain")
    if chain is None:
        st.error(t("chat.not_ready"))
        return

    with st.chat_message("assistant"), st.spinner(t("chat.thinking")):
        try:
            result = chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": SESSION_ID}},
            )
        except MissingAPIKeyError as exc:
            st.error(t("errors.missing_api_key", provider=str(exc)))
            return
        answer = str(result.get("answer", ""))
        st.markdown(answer)


# ------------------------------------------------------------- callbacks


def _on_init_rag(*, kind: SourceKind, uploaded: list[Any], url: str) -> None:
    if kind in FILE_KINDS and not uploaded:
        st.error(t("errors.no_source"))
        return
    if kind in {"site", "youtube"} and not url.strip():
        st.error(t("errors.no_source"))
        return

    try:
        with st.spinner(t("status.ingesting", kind=t(f"sidebar.source.{kind}"))):
            inputs: list[Any] | str = uploaded if kind in FILE_KINDS else url
            proxy = _proxy_from_env()
            docs = load_source(kind, inputs, locale=st.session_state["locale"], proxy=proxy)
            chunks = split_documents(docs)
            embeddings_provider: EmbeddingsProvider = st.session_state["embeddings_provider"]
            embeddings = get_embeddings(embeddings_provider)
            backend: VectorStoreBackend = st.session_state["vector_backend"]
            factory = VectorStoreFactory(
                backend=backend,
                persist=bool(st.session_state["persist_index"]),
                persist_dir=_persist_dir_for(backend),
            )
            vector_store: VectorStore = factory.build(
                chunks,
                embeddings,
                embeddings_fingerprint=fingerprint(embeddings_provider),
            )
            llm = get_llm(
                st.session_state["llm_provider"],
                model=st.session_state["llm_model"],
                temperature=float(st.session_state["temperature"]),
            )
            st.session_state["rag_chain"] = build_rag_chain(vector_store, llm)
            st.session_state["retriever_ready"] = True
    except MissingAPIKeyError as exc:
        st.error(t("errors.missing_api_key", provider=str(exc)))
    except InvalidYouTubeURLError:
        st.error(t("errors.invalid_url", kind="YouTube"))
    except TranscriptUnavailableError:
        st.error(t("errors.no_transcript"))
    except VideoUnavailableError:
        st.error(t("errors.video_unavailable"))
    except RequestBlockedError:
        st.error(t("errors.ip_blocked"))
    except AgeRestrictedError:
        st.error(t("errors.age_restricted"))
    except YouTubeLoaderError as exc:
        st.error(str(exc))
    except EmbeddingsMismatchError:
        st.warning(t("errors.embeddings_changed"))
    except Exception as exc:  # last-resort UI guard
        logger.exception("RAG initialization failed")
        st.error(str(exc))
    else:
        st.success(t("status.ready"))


def _on_clear_history() -> None:
    history = StreamlitChatMessageHistory(key=DEFAULT_HISTORY_KEY)
    history.clear()
    st.toast(t("status.cleared"))


def _proxy_from_env() -> dict[str, str] | None:
    http = os.getenv("YT_PROXY_HTTP", "")
    https = os.getenv("YT_PROXY_HTTPS", "")
    if not http and not https:
        return None
    return {"http": http, "https": https}


def _persist_dir_for(backend: VectorStoreBackend) -> str:
    return {"chroma": "./chroma_db", "faiss": "./faiss_index", "qdrant": "./qdrant_data"}[backend]


if __name__ == "__main__":
    main()
