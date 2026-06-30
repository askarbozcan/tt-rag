"""Visualize parser output against the sample files in this directory.

Run it to eyeball what each parser produces:

    python -m src.ingestion.parsers.test_files.tester
    # or limit/expand:
    python -m src.ingestion.parsers.test_files.tester --max-chars 0   # full text
    python -m src.ingestion.parsers.test_files.tester sample-tables.pdf
"""

import argparse
import sys
from pathlib import Path

# Allow running as a plain script (python .../tester.py) as well as -m.
# .../tt-rag/src/ingestion/parsers/test_files/tester.py -> parents[4] == repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ingestion.parsers._base import BaseParser, ParsedDocument  # noqa: E402
from src.ingestion.parsers.pymupdf_parser import PymupdfParser  # noqa: E402
from src.ingestion.parsers.python_docx_parser import PythonDocxParser  # noqa: E402

HERE = Path(__file__).resolve().parent

# Map file extension -> parser instance.
PARSERS: dict[str, BaseParser] = {
    ".pdf": PymupdfParser(),
    ".docx": PythonDocxParser(),
}


def _parser_for(path: Path) -> BaseParser | None:
    return PARSERS.get(path.suffix.lower())


def visualize(path: Path, max_chars: int) -> None:
    parser = _parser_for(path)
    print("\n" + "=" * 72)
    print(f"FILE   {path.name}  ({path.stat().st_size:,} bytes)")
    if parser is None:
        print(f"  (no parser registered for '{path.suffix}', skipping)")
        return
    print(f"PARSER {type(parser).__name__} v{getattr(parser, 'version', '?')}")
    print("=" * 72)

    doc: ParsedDocument = parser.parse(path.name, path.read_bytes())

    print(f"title       : {doc.title!r}")
    print(f"file_type   : {doc.file_type.name}")
    print(f"page_count  : {doc.page_count}")
    print(f"total chars : {len(doc.content):,}")
    print("metadata    :")
    for k, v in doc.metadata.items():
        if v not in ("", None):
            print(f"    {k}: {v!r}")

    for page in doc.pages:
        body = page.content
        truncated = ""
        if max_chars and len(body) > max_chars:
            truncated = f"  … (+{len(body) - max_chars:,} more chars)"
            body = body[:max_chars]
        print("\n" + "-" * 72)
        print(f"PAGE {page.page_number}  ({len(page.content):,} chars){truncated}")
        print("-" * 72)
        print(body if body.strip() else "  <empty>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "files",
        nargs="*",
        help="file names (relative to this dir) or paths; default: all samples here",
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="truncate each page to this many chars (0 = no limit)",
    )
    args = ap.parse_args()

    if args.files:
        targets = [Path(f) if Path(f).is_absolute() else HERE / f for f in args.files]
    else:
        targets = sorted(p for p in HERE.iterdir() if p.suffix.lower() in PARSERS)

    if not targets:
        print(f"No parseable sample files found in {HERE}")
        return

    for path in targets:
        if not path.exists():
            print(f"\n(missing: {path})")
            continue
        visualize(path, args.max_chars)


if __name__ == "__main__":
    main()
