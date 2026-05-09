from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    root = Path("runs")
    out = Path("paper_artifacts")
    out.mkdir(exist_ok=True)
    rows = []
    for path in root.rglob("*"):
        if path.is_file():
            rows.append({"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    with (out / "ARTIFACT_MANIFEST.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out / 'ARTIFACT_MANIFEST.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
