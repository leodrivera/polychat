"""History-aware RAG chain.

Wires:

    user_input + chat_history
          ▼  (condense prompt)
    standalone question
          ▼
    retriever  →  context docs
          ▼  (QA prompt)
    LLM answer

Message history is held by :class:`StreamlitChatMessageHistory`, which backs
onto ``streamlit.session_state`` so the chat survives reruns.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStore

from ..prompts.qa import condense_question_prompt, qa_prompt

DEFAULT_HISTORY_KEY = "chat_history"
DEFAULT_K = 4


HistoryFactory = Callable[[str], BaseChatMessageHistory]


def build_rag_chain(
    vector_store: VectorStore,
    llm: BaseChatModel,
    *,
    k: int = DEFAULT_K,
    history_key: str = DEFAULT_HISTORY_KEY,
    history_factory: HistoryFactory | None = None,
) -> RunnableWithMessageHistory:
    """Build the full RAG chain with history-aware retrieval.

    ``history_factory`` takes a session id and returns a chat history. Defaults
    to :class:`StreamlitChatMessageHistory`. Tests inject an in-memory history.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, condense_question_prompt()
    )
    document_chain = create_stuff_documents_chain(llm, qa_prompt())
    rag_chain: Runnable[dict[str, Any], dict[str, Any]] = create_retrieval_chain(
        history_aware_retriever, document_chain
    )

    factory = history_factory or _default_history_factory(history_key)

    return RunnableWithMessageHistory(
        rag_chain,
        factory,
        input_messages_key="input",
        history_messages_key=history_key,
        output_messages_key="answer",
    )


def _default_history_factory(history_key: str) -> HistoryFactory:
    def factory(_session_id: str) -> BaseChatMessageHistory:
        return StreamlitChatMessageHistory(key=history_key)

    return factory


__all__ = ["DEFAULT_HISTORY_KEY", "DEFAULT_K", "build_rag_chain"]
