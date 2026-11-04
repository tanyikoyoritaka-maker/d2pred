#!/usr/bin/env python3
"""Hash quarantined active v3 files and map them to their former paths."""
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent / "d2pred_rubbish/ablation_contamination_cleanup_20261103/quarantine"
OUTPUT = ROOT.parent / "d2pred_rubbish/ablation_contamination_cleanup_20261103/quarantine_manifest.csv"
MAP = (
    (BASE / "active_results/controlled_ablation_expert_exact_v3", ROOT / "results/controlled_ablation_expert_exact_v3"),
    (BASE / "active_code/analysis", ROOT / "analysis"),
    (BASE / "active_code/scripts", ROOT / "scripts"),
    (BASE / "active_code/tests", ROOT / "tests"),
    (BASE / "active_manifests", ROOT / "manifests"),
)
REASONS = {
    "active_results/controlled_ablation_expert_exact_v3": "Candidate metrics, checkpoints, and caches belong to INVALID_LEGACY_ABLATION; removed from formal result discovery.",
    "active_code/analysis": "Variant-only model code is retained outside the active import path pending historical review.",
    "active_code/scripts": "Variant-only preprocessor and result helper code is retained outside the active execution path.",
    "active_code/tests": "Legacy tests asserted the invalid v3 definitions and are retained as evidence, not as current validation.",
    "active_manifests": "V3 PASS audits, source bindings, and protocol text are retained for audit but no longer gate a formal run.",
}


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    rows = []
    for location, original_root in MAP:
        for path in sorted(item for item in location.rglob("*") if item.is_file()):
            relative = path.relative_to(location)
            original = original_root / relative
            category = str(location.relative_to(BASE))
            rows.append({
                "original_path": str(original),
                "original_sha256": digest(path),
                "quarantine_path": str(path),
                "quarantine_sha256": digest(path),
                "bytes": path.stat().st_size,
                "reason": REASONS[category],
            })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("records=%d bytes=%d output=%s" % (
        len(rows), sum(row["bytes"] for row in rows), OUTPUT))


if __name__ == "__main__":
    main()
