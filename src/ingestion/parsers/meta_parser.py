from ._base import BaseParser, FileType, ParsedDocument


class MetaParser(BaseParser):
    """Delegate parsing to a collection of parsers based on file type.

    The file type is detected from the file name's extension. Each delegate
    advertises the types it handles via :meth:`supported_filetypes`; the first
    parser registered for a given type wins.
    """

    def __init__(self, parsers: list[BaseParser]) -> None:
        self._by_filetype: dict[FileType, BaseParser] = {}
        for parser in parsers:
            for file_type in parser.supported_filetypes():
                self._by_filetype.setdefault(file_type, parser)

    def supported_filetypes(self) -> list[FileType]:
        return list(self._by_filetype)

    def parser_for(self, file_name: str) -> BaseParser:
        """Return the delegate that handles ``file_name``.

        Raises :class:`ValueError` if the extension is unrecognised or no
        registered parser supports its file type.
        """
        file_type = FileType.from_filename(file_name)
        if file_type is None:
            raise ValueError(f"Unsupported file extension: {file_name!r}")
        parser = self._by_filetype.get(file_type)
        if parser is None:
            raise ValueError(f"No parser registered for file type: {file_type.name}")
        return parser

    def parse(self, file_name: str, b: bytes) -> ParsedDocument:
        return self.parser_for(file_name).parse(file_name, b)
