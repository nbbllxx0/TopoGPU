from __future__ import annotations

import csv

import numpy as np

from topogpu.evidence import EvidenceBundle


def test_evidence_bundle_manifest_hashes(tmp_path) -> None:
    bundle = EvidenceBundle(tmp_path)
    bundle.write_history([{"iteration": 1, "compliance": 1.0}])
    bundle.write_summary({"status": "ok"})
    bundle.write_density(np.ones(3))
    bundle.write_render_metadata({"density_threshold": 0.5})
    manifest = bundle.write_manifest()
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {row["path"] for row in rows} >= {
        "history.csv",
        "summary.json",
        "rho_final.npy",
        "render_metadata.json",
    }
    assert all(len(row["sha256"]) == 64 for row in rows)
