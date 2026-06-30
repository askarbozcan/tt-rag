import pymupdf

from ._base import BaseParser, FileType, Page, ParsedDocument


class PymupdfParser(BaseParser):
    """Parse PDFs into per-page text using PyMuPDF (fitz)."""

    version = pymupdf.__version__

    def supported_filetypes(self) -> list[FileType]:
        return [FileType.PDF]

    def parse(self, file_name: str, b: bytes) -> ParsedDocument:
        with pymupdf.open(stream=b, filetype="pdf") as doc:
            pages = [
                Page(
                    content=page.get_text("text"), # type: ignore
                    page_number=page.number + 1, # type: ignore
                )
                for page in doc
            ]
            meta = doc.metadata or {} # type: ignore

        title: str = str(meta.get("title") or file_name) # type: ignore
        return ParsedDocument(
            pages=pages,
            file_type=FileType.PDF,
            metadata=dict(meta), # type: ignore
            title=title,
        )
