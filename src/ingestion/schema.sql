-- RAG ingestion schema (PostgreSQL + pgvector)
-- Embedding dimension: 1024
--
-- Model: documents 1 --- * chunks
-- A chunk's vector is produced under a specific ExtractionConfig (see
-- src/ingestion/configs.py). The config's SHA-256 fingerprint is stored on
-- each chunk so the same document can be re-ingested under different configs
-- and so ingestion is idempotent.

CREATE EXTENSION IF NOT EXISTS vector;
-- gen_random_uuid() is built in on PostgreSQL 13+. On older versions enable:
-- CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------------------
-- extraction_configs: provenance for how chunks were produced.
-- The PK is the config fingerprint (ExtractionConfig.fingerprint()), and the
-- full config is kept as JSONB so you can reproduce / audit any ingestion run.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS extraction_configs (
    config_hash   CHAR(64) PRIMARY KEY,        -- sha256 hex digest
    config        JSONB       NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- documents: one row per source artifact.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    uri           TEXT        NOT NULL UNIQUE,  -- canonical source location (s3://, file://, https://...)
    file_name     TEXT        NOT NULL,
    mime_type     TEXT,
    byte_size     BIGINT,
    content_hash  CHAR(64),                     -- sha256 of raw bytes; detect changed sources
    metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- chunks: retrievable units with their embeddings.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id            UUID        NOT NULL
                               REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index            INTEGER     NOT NULL,           -- ordinal within the document
    content                TEXT        NOT NULL,
    token_count            INTEGER,
    extraction_config_hash CHAR(64)    NOT NULL
                               REFERENCES extraction_configs (config_hash),
    embedding              VECTOR(1024) NOT NULL,
    metadata               JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Idempotency: a given position in a document under a given config is unique.
    -- Re-running ingestion with the same config is a no-op (ON CONFLICT DO NOTHING).
    UNIQUE (document_id, chunk_index, extraction_config_hash)
);

-- ---------------------------------------------------------------------------
-- Indices
-- ---------------------------------------------------------------------------

-- ANN search over embeddings. Cosine distance (<=>) assumes normalized
-- embeddings (ExtractionConfig.normalize_embeddings). If you store raw
-- (un-normalized) vectors and want L2/inner-product, swap the opclass for
-- vector_l2_ops / vector_ip_ops respectively.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Fetch / delete all chunks for a document.
CREATE INDEX IF NOT EXISTS chunks_document_id_idx
    ON chunks (document_id);

-- Filter retrieval to a specific extraction config (common when several
-- config versions coexist in the table).
CREATE INDEX IF NOT EXISTS chunks_config_hash_idx
    ON chunks (extraction_config_hash);
