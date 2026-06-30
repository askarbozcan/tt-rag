import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel

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

    



