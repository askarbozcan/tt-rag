"""REST API for the agentic RAG pipeline.

Exposes a single ``POST /query`` endpoint that runs the RAG agent to
completion and returns the final answer together with the chunks it
retrieved. No streaming — callers get the finished result in one response.

Configuration is read from the same environment variables the CLI in
``agentic_rag.py`` uses:

    DATABASE_DSN, API_KEY, OPENAI_BASE_URL, EMBEDDING_MODEL,
    EMBEDDING_SIZE, CHAT_MODEL, TOP_K (optional, default 15)

Run with::

    uv run uvicorn agents_and_graphs.api:app --reload
"""

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from src.inference.retrieval import RetrievedChunk, Retriever
from src.ingestion.embed import OpenAIEmbedder


@dataclass
class RagAgentDeps:
    """Per-request dependencies for the RAG agent.

    ``retrieved`` accumulates the chunks fetched during a run so the API can
    return them as sources alongside the final answer.
    """

    dsn: str
    embedder: OpenAIEmbedder
    top_k: int
    retrieved: list[RetrievedChunk] = field(default_factory=list)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_agent() -> Agent[RagAgentDeps, str]:
    """Construct the RAG agent and register its retrieval tool."""
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = _require_env("API_KEY")
    chat_model = _require_env("CHAT_MODEL")

    model = OpenAIChatModel(
        model_name=chat_model,
        provider=OpenAIProvider(base_url=base_url, api_key=api_key),
    )

    rag_agent = Agent(
        model=model,
        deps_type=RagAgentDeps,
        system_prompt=[
            "You are a helpful assistant. You will have access to a retrieval tool to search for information.",
            "Use it to answer the questions. Make sure you only use the information from the sources.",
        ],
    )

    @rag_agent.tool
    def retrieve(ctx: RunContext[RagAgentDeps], query: str) -> str:
        with Retriever.connect(ctx.deps.dsn, ctx.deps.embedder) as retriever:
            chunks = retriever.search(query, top_k=ctx.deps.top_k)
        ctx.deps.retrieved.extend(chunks)
        return "\n\n".join(c.content for c in chunks)

    return rag_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the agent and embedder once, at startup."""
    dsn = _require_env("DATABASE_DSN")
    api_key = _require_env("API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    embedding_model = _require_env("EMBEDDING_MODEL")
    embedding_size = int(_require_env("EMBEDDING_SIZE"))

    client = OpenAI(base_url=base_url, api_key=api_key)
    embedder = OpenAIEmbedder(
        client=client, model=embedding_model, dimensions=embedding_size
    )

    app.state.agent = build_agent()
    app.state.dsn = dsn
    app.state.embedder = embedder
    app.state.default_top_k = int(os.environ.get("TOP_K", "15"))
    yield


app = FastAPI(title="tt-rag agentic RAG API", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Chunks to retrieve per search. Defaults to the TOP_K env var.",
    )


class Source(BaseModel):
    content: str
    score: float
    file_name: str
    chunk_index: int
    page_number: int | None
    metadata: dict


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    top_k = request.top_k or app.state.default_top_k
    deps = RagAgentDeps(
        dsn=app.state.dsn, embedder=app.state.embedder, top_k=top_k
    )

    try:
        result = await app.state.agent.run(request.question, deps=deps)
    except Exception as exc:  # surface upstream/model failures as 502
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    sources = [
        Source(
            content=c.content,
            score=c.score,
            file_name=c.file_name,
            chunk_index=c.chunk_index,
            page_number=c.page_number,
            metadata=c.metadata,
        )
        for c in deps.retrieved
    ]
    return QueryResponse(answer=result.output, sources=sources)
