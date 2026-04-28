"""Minimal chain wiring test: build_rag_chain returns a runnable we can invoke."""

from __future__ import annotations

from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from polychat.rag.chain import build_rag_chain
from polychat.rag.vector_store import open_chroma_index


def test_build_rag_chain_produces_an_answer(fake_embeddings: Embeddings) -> None:
    docs = [
        Document(page_content="The capital of France is Paris.", metadata={"source": "geo.txt"}),
        Document(page_content="The Eiffel Tower is in Paris.", metadata={"source": "geo.txt"}),
    ]
    store = open_chroma_index(
        embeddings=fake_embeddings,
        embeddings_fingerprint="test:v1",
        persist_dir=None,
    )
    store.add_documents(docs)

    llm = FakeListChatModel(responses=["paris", "paris"])
    history_store: dict[str, InMemoryChatMessageHistory] = {}

    def history_factory(session_id: str) -> InMemoryChatMessageHistory:
        return history_store.setdefault(session_id, InMemoryChatMessageHistory())

    chain = build_rag_chain(store, llm, history_factory=history_factory)

    result: dict[str, Any] = chain.invoke(
        {"input": "What is the capital of France?"},
        config={"configurable": {"session_id": "t1"}},
    )
    assert "answer" in result
    assert isinstance(result["answer"], str)
