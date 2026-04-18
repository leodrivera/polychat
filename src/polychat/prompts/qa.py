"""QA and query-condensation prompts.

The QA prompt intentionally stays provider- and language-agnostic. The LLM
will answer in whatever language the user asks in; the UI ``t()`` layer is
independent of this.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

CONDENSE_QUESTION_SYSTEM = (
    "Given the chat history and the latest user question, rewrite the question "
    "as a standalone question that can be understood without the chat history. "
    "Do NOT answer the question; only reformulate it. Respond in the same "
    "language the user wrote in."
)

QA_SYSTEM = (
    "You are PolyChat, a helpful assistant that answers questions "
    "strictly based on the provided context from the user's documents, "
    "websites, or video transcripts.\n\n"
    "Rules:\n"
    "1. Answer ONLY from the context below. If the context does not contain "
    "the answer, say you don't know — do not invent facts.\n"
    "2. Reply in the same language as the user's question.\n"
    "3. Be concise by default; expand only when the user asks for detail.\n"
    "4. When useful, cite the source by referencing its filename or URL.\n\n"
    "Context:\n{context}"
)


def condense_question_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", CONDENSE_QUESTION_SYSTEM),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


def qa_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


__all__ = ["CONDENSE_QUESTION_SYSTEM", "QA_SYSTEM", "condense_question_prompt", "qa_prompt"]
