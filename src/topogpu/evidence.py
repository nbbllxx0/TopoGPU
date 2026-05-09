"""Evidence bundle and manifest helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(slots=True)
class EvidenceBundle:
    """Manifest-tracked run artifact writer."""

    root: Path
    rows: list[dict[str, Any]] = field(default_factory=list)

    def write_summary(self, summary: dict[str, Any]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.add(path, role="summary")
        return path

    def write_history(self, rows: list[dict[str, Any]]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "history.csv"
        fields = sorted({key for row in rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        self.add(path, role="history")
        return path

    def write_density(self, rho: np.ndarray) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "rho_final.npy"
        np.save(path, rho)
        self.add(path, role="density")
        return path

    def add(self, path: Path, role: str) -> None:
        resolved = path.resolve()
        self.rows.append(
            {
                "path": str(resolved.relative_to(self.root.resolve())),
                "role": role,
                "bytes": resolved.stat().st_size,
                "sha256": _sha256(resolved),
            }
        )

    def write_manifest(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "ARTIFACT_MANIFEST.csv"
        fields = ["path", "role", "bytes", "sha256"]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)
        return path

