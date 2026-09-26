"""Common immutable provenance schema (no method implementations)."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class BaselineContract:
    name: str
    paper: str
    source_url: str | None
    source_revision: str | None
    fidelity: str
    preserved_core: tuple[str, ...]
    static_framework_adaptations: tuple[str, ...]
    unresolved_gaps: tuple[str, ...]

    def as_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "paper": self.paper,
            "source_url": self.source_url,
            "source_revision": self.source_revision,
            "fidelity": self.fidelity,
            "preserved_core": list(self.preserved_core),
            "static_framework_adaptations": list(self.static_framework_adaptations),
            "unresolved_gaps": list(self.unresolved_gaps),
        }
