import hashlib
import json
from importlib.metadata import version
from typing import Any, Literal

from pydantic import BaseModel

from .parsers._base import BaseParser


class ExtractionConfig(BaseModel):
    """
        Extraction and ingestion configs
        Used for provenance and idempotency.
    """

    embedding_provider: str
    embedding_model: str
    embedding_size: int

    chonkie_ver: str
    chunking_method: Literal["recursive", "semantic"]
    chunking_method_params: dict[str, int | str]
    tokenizer: Literal["character", "gpt2"]

    parser: str
    parser_ver: str
    parser_params: dict[str, int | str] = {}

    @staticmethod
    def new(
        parser: BaseParser,
        embedding_provider: str,
        embedding_model: str,
        embedding_size: int,
        chunking_method: Literal["recursive", "semantic"],
        chunking_method_params: dict[str, int | str],
        tokenizer: Literal["character", "gpt2"],
    ) -> "ExtractionConfig":
        """Build a config for a given parser.

        The parser name/version vary by file type (PDF vs DOCX), so the config —
        and therefore its idempotency fingerprint — is built per file. The
        chonkie version is read from the installed package.
        """
        return ExtractionConfig(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_size=embedding_size,
            chonkie_ver=version("chonkie"),
            chunking_method=chunking_method,
            chunking_method_params=chunking_method_params,
            tokenizer=tokenizer,
            parser=type(parser).__name__,
            parser_ver=getattr(parser, "version", "unknown"),
        )

    def canonical_dict(self) -> dict[str, Any]:
        """Config as a plain dict in a stable, canonical form."""
        return self.model_dump(mode="json")

    def fingerprint(self) -> str:
        """
        Deterministic content hash of this config.

        Stable across processes and runs: keys are sorted and the JSON
        encoding is whitespace-free, so two configs that are semantically
        equal always produce the same fingerprint. Use it as an idempotency
        key for ingestion / provenance.
        """
        payload = json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    



