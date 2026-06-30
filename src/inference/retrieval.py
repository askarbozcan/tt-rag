"""Vector retrieval over the ingested ``chunks`` table.

The query is embedded with the same OpenAI-compatible embeddings API used at
ingestion time, then matched against stored chunk vectors by cosine distance
(``<=>``, the operator the HNSW index is built for). Keep the embedding model
and dimensionality identical to ingestion, or the vectors will not be
comparable.
"""

from dataclasses import dataclass

import psycopg2
from psycopg2.extensions import connection as Connection
from psycopg2.extras import RealDictCursor

from ..ingestion.db import _vector_literal # type: ignore
from ..ingestion.embed import OpenAIEmbedder


@dataclass
class RetrievedChunk:
    """A chunk returned by similarity search, with its relevance score."""

    content: str
    score: float                    # cosine similarity in [-1, 1]; higher is closer
    file_name: str
    chunk_index: int
    page_number: int | None
    metadata: dict[str, object]


class Retriever:
    """Embed a query and fetch the nearest chunks from PostgreSQL + pgvector."""

    def __init__(self, conn: Connection, embedder: OpenAIEmbedder) -> None:
        self._conn = conn
        self._embedder = embedder

    @classmethod
    def connect(cls, dsn: str, embedder: OpenAIEmbedder) -> "Retriever":
        """Open a connection from a libpq DSN and pair it with an embedder."""
        return cls(psycopg2.connect(dsn), embedder)

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> "Retriever":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # -- search --------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the ``top_k`` chunks most similar to ``query``."""
        [embedding] = self._embedder.embed_texts([query])
        literal = _vector_literal(embedding)

        with self._conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    c.content,
                    1 - (c.embedding <=> %s::vector) AS score,
                    c.chunk_index,
                    c.metadata,
                    d.file_name
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (literal, literal, top_k),
            )
            rows = cur.fetchall()

        return [
            RetrievedChunk(
                content=row["content"],
                score=float(row["score"]),
                file_name=row["file_name"],
                chunk_index=row["chunk_index"],
                page_number=row["metadata"].get("page_number"),
                metadata=row["metadata"],
            )
            for row in rows
        ]
