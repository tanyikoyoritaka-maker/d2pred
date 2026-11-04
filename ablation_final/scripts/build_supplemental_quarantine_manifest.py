#!/usr/bin/env python3
"""Hash the residual v3 figure artifacts and pre-cleanup audit snapshots."""
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
ARCHIVE_ROOT = WORKSPACE / "d2pred_rubbish/ablation_contamination_cleanup_20261103/supplemental_quarantine"
OUTPUT = WORKSPACE / "d2pred_rubbish/ablation_contamination_cleanup_20261103/supplemental_quarantine_manifest.csv"
SOURCE_ROOTS = {
    "active_figures": ROOT / "figures",
    "active_manifests": ROOT / "manifests",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    rows = []
    for archived in sorted(path for path in ARCHIVE_ROOT.rglob("*") if path.is_file()):
        relative = archived.relative_to(ARCHIVE_ROOT)
        category = relative.parts[0]
        source_relative = Path(*relative.parts[1:])
        if category == "active_manifests":
            source_relative = Path(source_relative.name.replace("_pre_cleanup", ""))
        source = SOURCE_ROOTS[category] / source_relative
        digest = sha256(archived)
        rows.append({
            "original_path": str(source),
            "original_sha256": digest,
            "archive_path": str(archived),
            "archive_sha256": digest,
            "bytes": archived.stat().st_size,
            "status": "ARCHIVED_READ_ONLY",
            "reason": "v3 candidate figure artifacts or pre-cleanup audit snapshot; retained outside active execution and result paths.",
        })
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"records={len(rows)} manifest={OUTPUT}")


if __name__ == "__main__":
    main()
