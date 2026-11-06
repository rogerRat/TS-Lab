"""Download the official GIFT-Eval dataset into the spec-owned data root."""

from __future__ import annotations

from pathlib import Path

from tsl.spec.ref_extraction import ReferenceExtractionSpecification


def fetch(spec: ReferenceExtractionSpecification, declaration_path: Path, data_root: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="Salesforce/GiftEval",
        repo_type="dataset",
        revision=spec.mapping["upstream"]["data_revision"],
        local_dir=str(data_root),
    )
