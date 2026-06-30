from dataclasses import dataclass
import os

from openai import OpenAI

from .chunk import Chunk
from .configs import ExtractionConfig

# OpenAI's embeddings endpoint accepts at most 2048 inputs per request.
_MAX_INPUTS_PER_REQUEST = 2048


@dataclass
class EmbeddedChunk:
    """A :class:`Chunk` paired with its embedding vector."""

    chunk: Chunk
    embedding: list[float]
    model: str


class OpenAIEmbedder:
    """Embed :class:`Chunk`s with OpenAI's embeddings API.

    Requests are batched and the chunk order is preserved, so the returned
    :class:`EmbeddedChunk`s line up one-to-one with the input chunks.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        dimensions: int,
        batch_size: int = _MAX_INPUTS_PER_REQUEST,
        
    ) -> None:
        if batch_size < 1 or batch_size > _MAX_INPUTS_PER_REQUEST:
            raise ValueError(
                f"batch_size must be in [1, {_MAX_INPUTS_PER_REQUEST}], got {batch_size}"
            )
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self._client = client

    @classmethod
    def from_config(cls, client: OpenAI, config: ExtractionConfig) -> "OpenAIEmbedder":
        """Build an embedder from an :class:`ExtractionConfig`.

        Reads ``embedding_model``, ``embedding_size`` and
        ``normalize_embeddings``.
        """

        return cls(
            client=client,
            model=config.embedding_model,
            dimensions=config.embedding_size
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed raw strings, preserving order."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):

            batch = texts[start : start + self.batch_size]
            kwargs: dict[str, object] = {"model": self.model, "input": batch}
            kwargs["dimensions"] = self.dimensions

            response = self._client.embeddings.create(**kwargs)  # type: ignore[arg-type]
            # The API guarantees data is returned in input order, but sort
            # defensively on 'index' so we never mis-align a vector to a chunk.
            for item in sorted(response.data, key=lambda d: d.index):
                vectors.append(item.embedding)
                
        return vectors

    def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed chunks, returning one :class:`EmbeddedChunk` per input chunk."""
        vectors = self.embed_texts([c.text for c in chunks])
        return [
            EmbeddedChunk(
                chunk=chunk,
                embedding=vector,
                model=self.model
            )
            for chunk, vector in zip(chunks, vectors)
        ]



if __name__ == "__main__":
    from .chunk import RecursiveDocumentChunker
    from .parsers._base import FileType, Page, ParsedDocument

    _EXAMPLE_TEXT = (
        "Retrieval-augmented generation (RAG) combines a retriever with a "
        "generator. The retriever finds relevant passages from a knowledge "
        "base. The generator then conditions its answer on those passages.\n\n"
        "Chunking is the step that turns raw documents into retrievable units."
    )

    doc = ParsedDocument(
        pages=[Page(content=_EXAMPLE_TEXT, page_number=1)],
        file_type=FileType.PDF,
        title="RAG embedding demo",
    )

    chunks = RecursiveDocumentChunker(chunk_size=256).chunk(doc)
    client = OpenAI(
        base_url="https://api.deepinfra.com/v1/",
        api_key=os.environ.get("API_KEY"),
    )

    embedder = OpenAIEmbedder(
        client=client,
        dimensions=1024,
        model="BAAI/bge-m3"
    )

    embedded = embedder.embed(chunks)

    print(f"{len(embedded)} chunks embedded with {embedder.model!r}\n")
    for ec in embedded:
        preview = ec.chunk.text[:60].replace("\n", " ")
        print(f"--- chunk {ec.chunk.index} | dim {len(ec.embedding)} ---")
        print(f"text:  {preview}...")
        print(f"vector: [{ec.embedding[0]:.4f}, {ec.embedding[1]:.4f}, ...]\n")