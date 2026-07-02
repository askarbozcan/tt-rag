from collections.abc import AsyncIterable
from dataclasses import dataclass

import click
from openai import OpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolReturnPart,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from src.inference.retrieval import RetrievedChunk, Retriever
from src.ingestion.embed import OpenAIEmbedder


def _truncate(text: str, limit: int = 800) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"… [+{len(text) - limit} chars]"


async def print_run_events(
    ctx: RunContext, event_stream: AsyncIterable[AgentStreamEvent]
) -> None:

    """Pretty-print tool calls and model text as they stream (reasoning is skipped)."""
    async for event in event_stream:
        if isinstance(event, FunctionToolCallEvent):
            print(f"\n┌─ tool call: {event.part.tool_name}")
            print(f"│  args: {event.part.args_as_json_str()}")
        elif isinstance(event, FunctionToolResultEvent):
            if isinstance(event.part, ToolReturnPart):
                print(f"└─ result: {_truncate(event.part.model_response_str())}")
            else:
                print(f"└─ retry: {event.part.model_response()}")
        elif isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
            print("\n╭─ model output " + "─" * 45)
            print(event.part.content, end="", flush=True)
        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
            print(event.delta.content_delta, end="", flush=True)
        elif isinstance(event, PartEndEvent) and isinstance(event.part, TextPart):
            print("\n╰" + "─" * 60)



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
    help="Chat model id used to generate answers scored by answer correctness.",
)
@click.option(
    "--top-k",
    envvar="TOP_K",
    default=15,
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
    help="Sampling temperature for answer generation.",
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

    model = OpenAIChatModel(
        model_name="Qwen/Qwen3.6-35B-A3B",
        provider=OpenAIProvider(
            base_url = base_url,
            api_key = api_key
        )
    )

    @dataclass
    class RagAgentDeps:
        dsn: str
        embedder: OpenAIEmbedder

    rag_agent = Agent(
        model=model,
        deps_type=RagAgentDeps,
        system_prompt=[
            "You are a helpful assistant. You will have access to a retrieval tool to search for information.",
            "Use it to answer the questions. Make sure you only use the information from the sources."
        ]
    )

    @rag_agent.tool
    def retrieve(ctx: RunContext[RagAgentDeps], query: str, top_k: int) -> str:
        dsn = ctx.deps.dsn
        embedder = ctx.deps.embedder
        with Retriever.connect(dsn, embedder) as retriever:
            chunks: list[RetrievedChunk] = retriever.search(query, top_k=top_k)

        chunk_texts = [c.content for c in chunks]

        return "\n\n".join(chunk_texts)

    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    embedder = OpenAIEmbedder(
        client = client,
        model = embedding_model,
        dimensions = embedding_size
    )

    test_deps = RagAgentDeps(
        dsn=dsn,
        embedder=embedder,
    )

    rag_agent.run_sync(
        "İstanbul'da Alaçatı yemekleri nerededir?",
        deps=test_deps,
        event_stream_handler=print_run_events,
    )

if __name__ == "__main__":
    main()


    