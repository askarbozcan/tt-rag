import io
from collections.abc import Iterator
from importlib.metadata import version

import docx
from docx.document import Document
from docx.oxml.ns import qn
from docx.table import Table

from ._base import BaseParser, FileType, Page, ParsedDocument

# WordprocessingML element tags (namespace-qualified).
_W_P = qn("w:p")
_W_TBL = qn("w:tbl")
_W_T = qn("w:t")
_W_TAB = qn("w:tab")
_W_BR = qn("w:br")
_W_TYPE = qn("w:type")
_W_PPR = qn("w:pPr")
_W_SECTPR = qn("w:sectPr")


class PythonDocxParser(BaseParser):
    """Parse DOCX files into text using python-docx.

    Word documents have no stable, render-independent page concept, so we can
    only split on breaks the author inserted explicitly: page breaks
    (``<w:br w:type="page"/>``) and section breaks (a paragraph-level
    ``<w:sectPr>``). Documents without such breaks come back as a single page.
    Paragraphs and tables are emitted in document order.
    """

    version = version("python-docx")

    def supported_filetypes(self) -> list[FileType]:
        return [FileType.DOCX]

    def parse(self, file_name: str, b: bytes) -> ParsedDocument:
        doc = docx.Document(io.BytesIO(b))

        # Accumulate text blocks into pages, starting a new page on each break.
        page_blocks: list[list[str]] = [[]]
        for kind, text in _iter_events(doc):
            if kind == "break":
                page_blocks.append([])
            elif text.strip():
                page_blocks[-1].append(text)

        pages = [
            Page(content="\n\n".join(blocks), page_number=i)
            for i, blocks in enumerate(
                (b for b in page_blocks if any(t.strip() for t in b)), start=1
            )
        ]
        if not pages:  # empty document
            pages = [Page(content="", page_number=1)]

        core = doc.core_properties
        meta: dict[str, object] = {
            "author": core.author or "",
            "created": core.created.isoformat() if core.created else "",
            "modified": core.modified.isoformat() if core.modified else "",
        }
        title = core.title or file_name

        return ParsedDocument(
            pages=pages,
            file_type=FileType.DOCX,
            metadata=meta,
            title=title,
        )


def _iter_events(doc: Document) -> Iterator[tuple[str, str]]:
    """Yield ``("text", str)`` and ``("break", "")`` events in document order.

    A ``break`` event is emitted at each explicit page break and at each
    section break.
    """
    for child in doc.element.body.iterchildren():  # type: ignore
        if child.tag == _W_P:  # type: ignore
            yield from _paragraph_events(child) # type: ignore
        elif child.tag == _W_TBL:  # type: ignore
            table = Table(child, doc) # type: ignore
            for row in table.rows:
                yield "text", "\t".join(cell.text for cell in row.cells)


def _paragraph_events(p) -> Iterator[tuple[str, str]]:  # type: ignore
    buf: list[str] = []
    for el in p.iter():  # type: ignore
        if el.tag == _W_T:  # type: ignore
            buf.append(el.text or "")  # type: ignore
        elif el.tag == _W_TAB:  # type: ignore
            buf.append("\t")
        elif el.tag == _W_BR and el.get(_W_TYPE) == "page":  # type: ignore
            yield "text", "".join(buf)
            buf = []
            yield "break", ""
    yield "text", "".join(buf)

    # A paragraph-level <w:sectPr> marks a section break after this paragraph.
    # (The final section's sectPr is a direct child of the body, not a
    # paragraph, so it correctly does not produce a trailing empty page.)
    pPr = p.find(_W_PPR)  # type: ignore
    if pPr is not None and pPr.find(_W_SECTPR) is not None:  # type: ignore
        yield "break", ""
