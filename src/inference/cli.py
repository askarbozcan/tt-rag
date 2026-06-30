"""Interactive RAG inference CLI with multi-turn conversations.

Each turn: the user's question is embedded and used to retrieve the most
relevant chunks from the database, those chunks are injected as context, and an
OpenAI-compatible chat model answers while the running conversation history is
preserved across turns.

Run it as a module::

    python -m src.inference.cli

Every option is also settable via an environment variable (see ``envvar=``
below), so the same invocation works from a ``.env`` / container config.

In-chat commands:

    /reset     start a new conversation (clears history)
    /sources   show the sources retrieved for the last answer
    /exit      quit (Ctrl-D / Ctrl-C also work)
"""

from dataclasses import dataclass

import click
from openai import OpenAI

from ..ingestion.embed import OpenAIEmbedder
from .retrieval import RetrievedChunk, Retriever

_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using the provided "
    "context retrieved from a document knowledge base. Ground your answers in "
    "that context and cite the source file names you used. If the context does "
    "not contain the answer, say so plainly instead of guessing."
)


@dataclass
class Message:
    """One turn of the conversation as stored in history (no injected context)."""

    role: str       # "user" or "assistant"
    content: str


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into a context block for the model."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        loc = c.file_name
        if c.page_number is not None:
            loc += f", p.{c.page_number}"
        blocks.append(f"[{i}] ({loc})\n{c.content}")
    return "\n\n".join(blocks)


def _build_request_messages(
    history: list[Message], context: str
) -> list[dict[str, str]]:
    """Assemble the messages sent to the chat API for the current turn.

    The full history is replayed for multi-turn coherence, but the retrieved
    context is attached only to the latest user turn so stale context from
    earlier turns does not accumulate.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for i, msg in enumerate(history):
        is_last_user = i == len(history) - 1 and msg.role == "user"
        if is_last_user:
            content = (
                f"Context:\n{context}\n\n"
                f"Question: {msg.content}"
                if context
                else msg.content
            )
            messages.append({"role": msg.role, "content": content})
        else:
            messages.append({"role": msg.role, "content": msg.content})
    return messages


@click.command()
@click.option(
    "--dsn",
    envvar="DATABASE_DSN",
    required=True,
    help="libpq DSN, e.g. postgresql://user:pass@host:5432/db",
)
@click.option(
    "--api-key",
    envvar="API_KEY",
    required=True,
    help="API key for the embeddings and chat endpoints.",
)
@click.option(
    "--base-url",
    envvar="OPENAI_BASE_URL",
    show_default=True,
    help="Base URL of the OpenAI-compatible API.",
)
@click.option(
    "--embedding-model",
    envvar="EMBEDDING_MODEL",
    required=True,
    help="Embedding model id (must match the model used at ingestion).",
)
@click.option(
    "--embedding-size",
    envvar="EMBEDDING_SIZE",
    type=int,
    required=True,
    help="Embedding dimensionality (must match the schema's VECTOR size).",
)
@click.option(
    "--chat-model",
    envvar="CHAT_MODEL",
    required=True,
    help="Chat completion model id used to generate answers.",
)
@click.option(
    "--top-k",
    envvar="TOP_K",
    default=5,
    show_default=True,
    type=int,
    help="Number of chunks to retrieve per question.",
)
@click.option(
    "--temperature",
    envvar="TEMPERATURE",
    default=0.2,
    show_default=True,
    type=float,
    help="Sampling temperature for the chat model.",
)
def main(
    dsn: str,
    api_key: str,
    base_url: str,
    embedding_model: str,
    embedding_size: int,
    chat_model: str,
    top_k: int,
    temperature: float,
) -> None:
    """Start an interactive multi-turn RAG chat session."""
    client = OpenAI(base_url=base_url, api_key=api_key)
    embedder = OpenAIEmbedder(
        client=client, model=embedding_model, dimensions=embedding_size
    )

    history: list[Message] = []
    last_sources: list[RetrievedChunk] = []

    click.echo("RAG chat ready. Type a question, or /reset, /sources, /exit.")

    with Retriever.connect(dsn, embedder) as retriever:
        while True:
            try:
                question = click.prompt("you", prompt_suffix="> ").strip()
            except (EOFError, click.Abort):
                click.echo()
                break

            if not question:
                continue
            if question == "/exit":
                break
            if question == "/reset":
                history.clear()
                last_sources = []
                click.echo("Conversation reset.")
                continue
            if question == "/sources":
                _echo_sources(last_sources)
                continue

            chunks = retriever.search(question, top_k=top_k)
            last_sources = chunks
            history.append(Message(role="user", content=question))

            messages = _build_request_messages(history, _format_context(chunks))
            response = client.chat.completions.create(
                model=chat_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
            )
            answer = response.choices[0].message.content or ""
            history.append(Message(role="assistant", content=answer))

            click.echo()
            click.secho(answer, fg="green")
            click.echo()


def _echo_sources(sources: list[RetrievedChunk]) -> None:
    """Print the sources retrieved for the most recent answer."""
    if not sources:
        click.echo("No sources yet.")
        return
    click.echo("Sources for the last answer:")
    for i, c in enumerate(sources, start=1):
        loc = c.file_name
        if c.page_number is not None:
            loc += f", p.{c.page_number}"
        click.echo(f"  [{i}] {loc}  (score {c.score:.3f})")


if __name__ == "__main__":
    main()
