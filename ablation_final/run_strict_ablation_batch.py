#!/usr/bin/env python3
"""Run the complete strict ablation matrix with per-run provenance."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
ABLATION = ROOT / "ablation_final"
OUTPUT = ABLATION / "outputs/strict_ablation_v2"
MANIFEST_PATH = OUTPUT / "batch_run_manifest.json"
PROTOCOL_PATH = ABLATION / "protocol/ablation_manifest_v2.json"
TRAINER = ABLATION / "train_strict_ablation.py"
DATA = ROOT / "data/industrial/DI_df.csv"
DATASET_NAMES = ("ACSim", "ACCom", "ACAll")
VARIANT_NAMES = (
    "no_wiener", "no_stl", "no_autoencoder_fusion", "no_dwc",
    "no_edc", "no_decomposition", "no_fusion", "no_aep",
)
SEEDS = (17, 42, 73)
CODE_FILES = (
    "analysis/expert_exact_d2pred.py",
    "analysis/strict_ablation.py",
    "analysis/theory_aligned_decomposition.py",
    "analysis/causal_decomposition.py",
    "ablation_final/strict_pipeline.py",
    "ablation_final/train_strict_ablation.py",
    "ablation_final/run_strict_ablation_batch.py",
    "ablation_final/evaluate_strict_ablation.py",
    "run_ac_nested_gru_rerun.py",
    "run_stage_aware_aefusion_ac.py",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_manifest(value: dict) -> None:
    temporary = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, MANIFEST_PATH)


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(OUTPUT.resolve()))


def main() -> int:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "VERIFIED_COMPLETED" or protocol.get("execution_enabled") is not True:
        raise RuntimeError("Strict ablation protocol is not approved for execution")
    if digest(DATA) != protocol["data"]["sha256"]:
        raise RuntimeError("Canonical DI source hash differs from the approved protocol")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    batch_id = f"strict_v2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    fold_hashes = {
        dataset: digest(Path(protocol["data"]["fold_manifests"][dataset]["path"]))
        for dataset in DATASET_NAMES
    }
    code_fingerprints = {name: digest(ROOT / name) for name in CODE_FILES}
    source_commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    diff_result = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=ROOT, capture_output=True, check=False
    )
    source_commit = source_commit_result.stdout.strip() if source_commit_result.returncode == 0 else None
    diff_sha256 = hashlib.sha256(diff_result.stdout).hexdigest() if diff_result.returncode == 0 else None
    records = []
    for dataset in DATASET_NAMES:
        for variant in VARIANT_NAMES:
            for fold in range(3):
                for seed in SEEDS:
                    records.append({
                        "run_id": f"{dataset}__{variant}__fold_{fold}__seed_{seed}",
                        "dataset": dataset,
                        "variant": variant,
                        "outer_fold": fold,
                        "seed": seed,
                        "protocol_id": protocol["protocol_id"],
                        "status": "PENDING",
                        "started_at_utc": None,
                        "completed_at_utc": None,
                        "command": None,
                        "exit_code": None,
                        "data_and_fold_manifest_sha256": {
                            "di_source": protocol["data"]["sha256"],
                            "fold_manifest": fold_hashes[dataset],
                        },
                        "code_fingerprints": code_fingerprints,
                        "artifacts": {},
                    })

    manifest = {
        "record_type": "batch_run_manifest",
        "protocol_id": protocol["protocol_id"],
        "protocol_status_at_execution": protocol["status"],
        "protocol_manifest_sha256": digest(PROTOCOL_PATH),
        "config_matrix_sha256": digest(ABLATION / "protocol/ablation_config_matrix_v2.csv"),
        "batch_run_id": batch_id,
        "started_at_utc": now(),
        "completed_at_utc": None,
        "source_commit": source_commit,
        "working_tree_diff_sha256": diff_sha256,
        "code_fingerprints": code_fingerprints,
        "data_source_manifest_path": "data/data_source_manifest.json",
        "data_source_manifest_sha256": digest(OUTPUT / "data/data_source_manifest.json"),
        "datasets": list(DATASET_NAMES),
        "variants": list(VARIANT_NAMES),
        "outer_folds": 3,
        "seeds": list(SEEDS),
        "expected_run_count": 216,
        "run_records": records,
        "metrics_manifest": "evaluation/metrics_manifest.json",
        "batch_status": "RUNNING",
    }
    write_manifest(manifest)

    for record in records:
        run_dir = OUTPUT / record["dataset"] / record["variant"] / f"outer_fold_{record['outer_fold']}" / f"seed_{record['seed']}"
        run_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = run_dir / "training_stdout.log"
        command = [
            sys.executable, str(TRAINER), "--execute",
            "--dataset", record["dataset"], "--variant", record["variant"],
            "--fold", str(record["outer_fold"]), "--seed", str(record["seed"]),
            "--output-root", str(OUTPUT),
        ]
        record["command"] = {"argv": command, "cwd": str(ROOT)}
        record["started_at_utc"] = now()
        record["status"] = "RUNNING"
        write_manifest(manifest)
        with stdout_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        record["completed_at_utc"] = now()
        record["exit_code"] = completed.returncode
        artifact_names = {
            "checkpoint": run_dir / "best_checkpoint.pt",
            "training_log": run_dir / "training_log.csv",
            "validation_selection": run_dir / "metadata.json",
            "validation_predictions": run_dir / "validation_predictions.csv",
            "test_predictions": run_dir / "test_predictions.csv",
            "stdout_log": stdout_path,
        }
        record["artifacts"] = {
            key: {"path": rel(path), "sha256": digest(path)}
            for key, path in artifact_names.items() if path.is_file()
        }
        record["status"] = "COMPLETED" if completed.returncode == 0 else "FAILED"
        write_manifest(manifest)
        if completed.returncode != 0:
            manifest["completed_at_utc"] = now()
            manifest["batch_status"] = "FAILED"
            write_manifest(manifest)
            return completed.returncode

    manifest["completed_at_utc"] = now()
    manifest["batch_status"] = "COMPLETED" if all(r["status"] == "COMPLETED" for r in records) else "INCOMPLETE"
    write_manifest(manifest)
    return 0 if manifest["batch_status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
