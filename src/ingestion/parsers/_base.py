import os
from dataclasses import dataclass, field
from typing import Protocol
from enum import Enum

class FileType(Enum):
    PDF = 1
    DOCX = 2

    @classmethod
    def from_filename(cls, file_name: str) -> "FileType | None":
        """Detect the file type from a file name's extension, or None if unknown."""
        ext = os.path.splitext(file_name)[1].lower().lstrip(".")
        return _EXTENSION_TO_FILETYPE.get(ext)


_EXTENSION_TO_FILETYPE: dict[str, FileType] = {
    "pdf": FileType.PDF,
    "docx": FileType.DOCX,
}

@dataclass
class Page:
    content: str
    page_number: int                                    # 1-based; slide number for PPTX
    metadata: dict[str, object] = field(default_factory=dict)

@dataclass
class ParsedDocument:
    pages: list[Page]
    file_type: FileType
    metadata: dict[str, object] = field(default_factory=dict)
    title: str | None = None

    @property
    def content(self) -> str:
        """Flat view of the whole document, for callers that don't need page structure."""
        return "\n\n".join(p.content for p in self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)


class BaseParser(Protocol):
    def supported_filetypes(self) -> list[FileType]:
        ...

    def parse(self, file_name: str, b: bytes) -> ParsedDocument:
        ...