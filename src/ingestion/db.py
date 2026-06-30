from pathlib import Path
from typing import Iterable

import psycopg2
from psycopg2.extensions import connection as Connection
from psycopg2.extras import Json, execute_values

from .configs import ExtractionConfig
from .embed import EmbeddedChunk

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _vector_literal(embedding: Iterable[float]) -> str:
    """Render an embedding as a pgvector text literal, e.g. ``[0.1,0.2,0.3]``.

    Keeps the dependency surface to plain psycopg2: the literal is passed as a
    parameter and cast with ``%s::vector`` at insert time, so the ``pgvector``
    Python package is not required.
    """
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


class IngestionDB:
    """Persist parsed/chunked/embedded documents to PostgreSQL (+ pgvector).

    A single :meth:`ingest` call writes one document and its chunks inside one
    transaction. Ingestion is idempotent: re-running with the same
    :class:`ExtractionConfig` re-uses the document row (by ``uri``) and skips
    chunks that already exist for ``(document_id, chunk_index, config_hash)``.
    """

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    @classmethod
    def connect(cls, dsn: str) -> "IngestionDB":
        """Open a connection from a libpq DSN (``postgresql://user:pass@host/db``)."""
        return cls(psycopg2.connect(dsn))

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> "IngestionDB":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    # -- schema --------------------------------------------------------------

    # Tables owned by this schema, ordered so that DROP respects FK dependencies.
    _TABLES = ("chunks", "documents", "extraction_configs")

    def init_schema(self, schema_path: Path = _SCHEMA_PATH) -> None:
        """Create the extension, tables and indices from ``schema.sql``."""
        sql = schema_path.read_text()
        with self._conn, self._conn.cursor() as cur:
            cur.execute(sql)

    def tables_exist(self) -> bool:
        """True if any of the schema's tables already exist in the database."""
        with self._conn, self._conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = ANY(%s))",
                (list(self._TABLES),),
            )
            return bool(cur.fetchone()[0])

    def drop_schema(self) -> None:
        """Drop the schema's tables (and dependent objects) if they exist."""
        with self._conn, self._conn.cursor() as cur:
            cur.execute(
                "DROP TABLE IF EXISTS "
                + ", ".join(self._TABLES)
                + " CASCADE"
            )

    # -- writes --------------------------------------------------------------

    def upsert_config(self, config: ExtractionConfig) -> str:
        """Store the config keyed by its fingerprint; return that fingerprint."""
        config_hash = config.fingerprint()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO extraction_configs (config_hash, config)
                VALUES (%s, %s)
                ON CONFLICT (config_hash) DO NOTHING
                """,
                (config_hash, Json(config.canonical_dict())),
            )
        return config_hash

    def upsert_document(
        self,
        uri: str,
        file_name: str,
        mime_type: str | None = None,
        byte_size: int | None = None,
        content_hash: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> str:
        """Insert or refresh a document row (keyed by ``uri``); return its UUID."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents
                    (uri, file_name, mime_type, byte_size, content_hash, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (uri) DO UPDATE SET
                    file_name    = EXCLUDED.file_name,
                    mime_type    = EXCLUDED.mime_type,
                    byte_size    = EXCLUDED.byte_size,
                    content_hash = EXCLUDED.content_hash,
                    metadata     = EXCLUDED.metadata,
                    updated_at   = now()
                RETURNING id
                """,
                (
                    uri,
                    file_name,
                    mime_type,
                    byte_size,
                    content_hash,
                    Json(metadata or {}),
                ),
            )
            return cur.fetchone()[0]

    def insert_chunks(
        self,
        document_id: str,
        config_hash: str,
        embedded: list[EmbeddedChunk],
    ) -> int:
        """Bulk-insert embedded chunks; return the number actually written.

        Existing ``(document_id, chunk_index, config_hash)`` rows are left
        untouched (``ON CONFLICT DO NOTHING``).
        """
        if not embedded:
            return 0

        rows = [
            (
                document_id,
                ec.chunk.index,
                ec.chunk.text,
                ec.chunk.token_count,
                config_hash,
                _vector_literal(ec.embedding),
                Json(
                    {
                        **ec.chunk.metadata,
                        "page_number": ec.chunk.page_number,
                        "start_index": ec.chunk.start_index,
                        "end_index": ec.chunk.end_index,
                        "embedding_model": ec.model,
                    }
                ),
            )
            for ec in embedded
        ]

        with self._conn.cursor() as cur:
            inserted = execute_values(
                cur,
                """
                INSERT INTO chunks
                    (document_id, chunk_index, content, token_count,
                     extraction_config_hash, embedding, metadata)
                VALUES %s
                ON CONFLICT (document_id, chunk_index, extraction_config_hash)
                DO NOTHING
                RETURNING 1
                """,
                rows,
                template="(%s, %s, %s, %s, %s, %s::vector, %s)",
                fetch=True,
            )
        return len(inserted)

    # -- orchestration -------------------------------------------------------

    def ingest(
        self,
        uri: str,
        file_name: str,
        embedded: list[EmbeddedChunk],
        config: ExtractionConfig,
        mime_type: str | None = None,
        byte_size: int | None = None,
        content_hash: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> tuple[str, int]:
        """Write a document and its chunks in one transaction.

        Returns ``(document_id, chunks_inserted)``. Rolls back on any error.
        """
        try:
            config_hash = self.upsert_config(config)
            document_id = self.upsert_document(
                uri,
                file_name,
                mime_type=mime_type,
                byte_size=byte_size,
                content_hash=content_hash,
                metadata=metadata,
            )
            inserted = self.insert_chunks(document_id, config_hash, embedded)
        except Exception:
            self._conn.rollback()
            raise
        self._conn.commit()
        return document_id, inserted