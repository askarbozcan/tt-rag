from dataclasses import dataclass, field

from chonkie import RecursiveChunker

from .configs import ExtractionConfig
from .parsers._base import ParsedDocument


@dataclass
class Chunk:
    """A contiguous slice of a :class:`ParsedDocument`, ready for embedding."""

    text: str
    index: int                                          # 0-based ordinal within the document
    page_number: int                                    # source page (1-based)
    start_index: int                                    # char offset within the source page
    end_index: int                                      # char offset within the source page
    token_count: int
    metadata: dict[str, object] = field(default_factory=dict)


class RecursiveDocumentChunker:
    """Chunk :class:`ParsedDocument`s with chonkie's :class:`RecursiveChunker`.

    Each page is chunked independently so every resulting chunk keeps the page
    number it came from; offsets are relative to that page's content.
    """

    def __init__(
        self,
        tokenizer: str = "character",
        chunk_size: int = 2048,
        min_characters_per_chunk: int = 24,
    ) -> None:
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.min_characters_per_chunk = min_characters_per_chunk
        self._chunker = RecursiveChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            min_characters_per_chunk=min_characters_per_chunk,
        )

    @classmethod
    def from_config(cls, config: ExtractionConfig) -> "RecursiveDocumentChunker":
        """Build a chunker from an :class:`ExtractionConfig`.

        Reads ``chunk_size`` and ``min_characters_per_chunk`` from
        ``config.chunking_method_params`` when present.
        """
        if config.chunking_method != "recursive":
            raise ValueError(
                f"RecursiveDocumentChunker requires chunking_method='recursive', "
                f"got {config.chunking_method!r}"
            )
        params = config.chunking_method_params
        kwargs: dict[str, object] = {"tokenizer": config.tokenizer}
        if "chunk_size" in params:
            kwargs["chunk_size"] = int(params["chunk_size"])
        if "min_characters_per_chunk" in params:
            kwargs["min_characters_per_chunk"] = int(params["min_characters_per_chunk"])
        return cls(**kwargs)  # type: ignore[arg-type]

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        for page in document.pages:
            if not page.content.strip():
                continue
            for c in self._chunker.chunk(page.content):
                chunks.append(
                    Chunk(
                        text=c.text,
                        index=index,
                        page_number=page.page_number,
                        start_index=c.start_index,
                        end_index=c.end_index,
                        token_count=c.token_count,
                        metadata=dict(page.metadata),
                    )
                )
                index += 1
        return chunks


if __name__ == "__main__":
    from .parsers._base import FileType, Page

    _EXAMPLE_TEXT = (
        "Retrieval-augmented generation (RAG) combines a retriever with a "
        "generator. The retriever finds relevant passages from a knowledge "
        "base. The generator then conditions its answer on those passages.\n\n"
        "Chunking is the step that turns raw documents into retrievable units. "
        "If chunks are too large, retrieval becomes imprecise and you waste "
        "context. If chunks are too small, you lose the surrounding meaning "
        "needed to answer a question.\n\n"
        "A recursive chunker splits text along a hierarchy of delimiters: "
        "paragraphs first, then sentences, then words. It keeps splitting "
        "until each piece fits within the configured chunk size. This tends "
        "to preserve natural boundaries better than a fixed-width window."
    )

    doc = ParsedDocument(
        pages=[Page(content=_EXAMPLE_TEXT, page_number=1)],
        file_type=FileType.PDF,
        title="RAG chunking demo",
    )

    chunker = RecursiveDocumentChunker(chunk_size=256, min_characters_per_chunk=24)
    chunks = chunker.chunk(doc)

    print(f"{len(chunks)} chunks (chunk_size={chunker.chunk_size}, "
          f"tokenizer={chunker.tokenizer!r})\n")
    for c in chunks:
        print(f"--- chunk {c.index} | page {c.page_number} | "
              f"chars [{c.start_index}:{c.end_index}] | {c.token_count} tokens ---")
        print(c.text)
        print()
