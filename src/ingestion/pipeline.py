"""End-to-end ingestion pipeline: directory of files -> Postgres + pgvector.

For each source file the pipeline:

1. parses it into a :class:`ParsedDocument` (parser chosen by extension),
2. chunks every page with chonkie's recursive chunker,
3. embeds the chunks with an OpenAI-compatible embeddings API, and
4. writes the document and chunks to the database under a fingerprinted
   :class:`ExtractionConfig` (so re-running is idempotent).

Run it as a module::

    python -m src.ingestion.pipeline --source-dir ./docs

Every option is also settable via an environment variable (see ``envvar=``
below), so the same invocation works from a ``.env`` / container config.
"""

import hashlib
import mimetypes
from pathlib import Path

import click
from openai import OpenAI



from .chunk import RecursiveDocumentChunker
from .configs import ExtractionConfig
from .db import IngestionDB
from .embed import OpenAIEmbedder

from .parsers._base import ParsedDocument
from .parsers.meta_parser import MetaParser
from .parsers.pymupdf_parser import PymupdfParser
from .parsers.python_docx_parser import PythonDocxParser

# Extensions we know how to parse; used to discover files in the source dir.
_SUPPORTED_SUFFIXES = {".pdf", ".docx"}


def _build_meta_parser() -> MetaParser:
    return MetaParser([PymupdfParser(), PythonDocxParser()])


def _discover_files(source_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
    )


@click.command()
@click.option(
    "--source-dir",
    envvar="SOURCE_DIR",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Local directory of source files to ingest (recursive).",
)
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
    help="API key for the embeddings endpoint.",
)
@click.option(
    "--base-url",
    envvar="OPENAI_BASE_URL",
    show_default=True,
    help="Base URL of the OpenAI-compatible embeddings API.",
)
@click.option(
    "--embedding-provider",
    envvar="EMBEDDING_PROVIDER",
    show_default=True,
    help="Provider name recorded in the extraction config.",
)
@click.option(
    "--embedding-model",
    envvar="EMBEDDING_MODEL",
    show_default=True,
    help="Embedding model id.",
)
@click.option(
    "--embedding-size",
    envvar="EMBEDDING_SIZE",
    type=int,
    help="Embedding dimensionality (must match the schema's VECTOR size).",
)
@click.option(
    "--chunk-size",
    envvar="CHUNK_SIZE",
    default=8123,
    show_default=True,
    type=int,
    help="Target chunk size for the recursive chunker.",
)
@click.option(
    "--min-characters-per-chunk",
    envvar="MIN_CHARACTERS_PER_CHUNK",
    default=24,
    show_default=True,
    type=int,
    help="Minimum characters per chunk.",
)
@click.option(
    "--tokenizer",
    envvar="TOKENIZER",
    default="character",
    show_default=True,
    type=click.Choice(["character", "gpt2"]),
    help="Tokenizer used to measure chunk size.",
)
@click.option(
    "--overwrite",
    envvar="OVERWRITE",
    is_flag=True,
    default=False,
    help="Drop and recreate existing tables without prompting.",
)
def main(
    source_dir: Path,
    dsn: str,
    api_key: str,
    base_url: str,
    embedding_provider: str,
    embedding_model: str,
    embedding_size: int,
    chunk_size: int,
    min_characters_per_chunk: int,
    tokenizer: str,
    overwrite: bool,
) -> None:
    """Parse, chunk, embed and load a directory of files into the database."""
    files: list[Path] = _discover_files(source_dir)
    if not files:
        raise click.ClickException(
            f"No supported files ({', '.join(sorted(_SUPPORTED_SUFFIXES))}) "
            f"found under {source_dir}"
        )
    click.echo(f"Found {len(files)} file(s) to ingest under {source_dir}")

    meta_parser = _build_meta_parser()
    chunker = RecursiveDocumentChunker(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        min_characters_per_chunk=min_characters_per_chunk,
    )
    client = OpenAI(base_url=base_url, api_key=api_key)
    embedder = OpenAIEmbedder(
        client=client, model=embedding_model, dimensions=embedding_size
    )
    chunking_method_params: dict[str, int | str] = {
        "chunk_size": chunk_size,
        "min_characters_per_chunk": min_characters_per_chunk,
    }

    with IngestionDB.connect(dsn) as db:
        # Ensure the schema exists, prompting before dropping existing tables
        # (--overwrite skips the prompt and drops unconditionally).
        if db.tables_exist():
            drop = overwrite or click.confirm(
                "Ingestion tables already exist. Drop and recreate them "
                "(this deletes all existing documents and chunks)?",
                default=False,
            )
            if drop:
                click.echo("Dropping existing tables...")
                db.drop_schema()
            else:
                click.echo("Keeping existing tables.")
        db.init_schema()

        total_chunks = 0
        for path in files:
            try:
                parser = meta_parser.parser_for(path.name)
            except ValueError as exc:
                click.echo(f"  skip {path.name}: {exc}")
                continue

            raw = path.read_bytes()
            document: ParsedDocument = parser.parse(path.name, raw)
            chunks = chunker.chunk(document)
            if not chunks:
                click.echo(f"  {path.name}: no chunks produced, skipping")
                continue
            embedded = embedder.embed(chunks)

            config = ExtractionConfig.new(
                parser,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_size=embedding_size,
                chunking_method="recursive",
                chunking_method_params=chunking_method_params,
                tokenizer=tokenizer,  # type: ignore[arg-type]
            )
            mime_type = mimetypes.guess_type(path.name)[0]
            metadata = {**document.metadata, "title": document.title}

            _, inserted = db.ingest(
                uri=path.resolve().as_uri(),
                file_name=path.name,
                embedded=embedded,
                config=config,
                mime_type=mime_type,
                byte_size=len(raw),
                content_hash=hashlib.sha256(raw).hexdigest(),
                metadata=metadata,
            )
            total_chunks += inserted
            click.echo(
                f"  {path.name}: {len(embedded)} chunk(s) embedded, "
                f"{inserted} inserted"
            )

    click.echo(f"Done. {total_chunks} new chunk(s) written.")


if __name__ == "__main__":
    main()
