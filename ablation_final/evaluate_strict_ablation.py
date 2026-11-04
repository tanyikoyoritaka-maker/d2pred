#!/usr/bin/env python3
"""Evaluate strict-v2 per-run predictions and record the exact input set."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "ablation_final/outputs/strict_ablation_v2"
BATCH_PATH = OUTPUT / "batch_run_manifest.json"
EVALUATION = OUTPUT / "evaluation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    batch = json.loads(BATCH_PATH.read_text(encoding="utf-8"))
    completed = [run for run in batch["run_records"] if run["status"] == "COMPLETED"]
    if batch["batch_status"] != "COMPLETED" or len(completed) != 216:
        raise RuntimeError(f"Expected 216 completed runs, found {len(completed)}")

    input_records = []
    test_frames = []
    validation_frames = []
    for run in completed:
        item = run["artifacts"].get("test_predictions")
        if item is None:
            raise RuntimeError(f"Missing test prediction for {run['run_id']}")
        path = OUTPUT / item["path"]
        actual_hash = sha256(path)
        if actual_hash != item["sha256"]:
            raise RuntimeError(f"Prediction checksum mismatch: {path}")
        input_records.append({
            "run_id": run["run_id"],
            "dataset": run["dataset"],
            "variant": run["variant"],
            "outer_fold": run["outer_fold"],
            "seed": run["seed"],
            "path": item["path"],
            "sha256": actual_hash,
        })
        prediction = pd.read_csv(path)
        prediction["dataset"] = run["dataset"]
        prediction["variant"] = run["variant"]
        test_frames.append(prediction)
        validation_item = run["artifacts"].get("validation_predictions")
        validation_path = OUTPUT / validation_item["path"]
        if sha256(validation_path) != validation_item["sha256"]:
            raise RuntimeError(f"Validation prediction checksum mismatch: {validation_path}")
        validation = pd.read_csv(validation_path)
        validation["dataset"] = run["dataset"]
        validation["variant"] = run["variant"]
        validation_frames.append(validation)

    full_records = []
    comparison_root = ROOT / "results/frozen_protocol_final/comparison"
    for dataset in batch["datasets"]:
        full_path = comparison_root / dataset / "test_predictions.csv"
        full_hash = sha256(full_path)
        input_records.append({
            "run_id": f"{dataset}__full__frozen_reference",
            "dataset": dataset,
            "variant": "full",
            "path": str(full_path.relative_to(ROOT)),
            "sha256": full_hash,
            "source": "frozen_primary_comparison",
        })
        full = pd.read_csv(full_path)
        full["dataset"] = dataset
        full["variant"] = "full"
        test_frames.append(full)
        validation_path = comparison_root / dataset / "validation_predictions.csv"
        full_validation = pd.read_csv(validation_path)
        full_validation["dataset"] = dataset
        full_validation["variant"] = "full"
        validation_frames.append(full_validation)

    predictions = pd.concat(test_frames, ignore_index=True)
    validations = pd.concat(validation_frames, ignore_index=True)
    run_metrics = []
    keys = ["dataset", "variant", "outer_fold", "seed", "unit"]
    for key, part in predictions.groupby(keys, sort=True):
        errors = part["predicted_rul_cycles"].to_numpy(float) - part["true_rul_cycles"].to_numpy(float)
        absolute = np.abs(errors)
        run_metrics.append(dict(zip(keys, key)) | {
            "RMSE": float(np.sqrt(np.mean(errors ** 2))),
            "MAE": float(np.mean(absolute)),
            "P90AE": float(np.percentile(absolute, 90)),
        })
    per_device_seed = pd.DataFrame(run_metrics)
    per_device = per_device_seed.groupby(["dataset", "variant", "unit"], as_index=False)[
        ["RMSE", "MAE", "P90AE"]
    ].mean()
    records = []
    for (dataset, variant), part in per_device.groupby(["dataset", "variant"], sort=True):
        records.append({
            "dataset": dataset,
            "variant": variant,
            "num_devices": int(part.unit.nunique()),
            "rmse_cycles": float(part.RMSE.mean()),
            "mae_cycles": float(part.MAE.mean()),
            "p90ae_cycles": float(part.P90AE.mean()),
            "outer_folds": 3,
            "seeds": "17;42;73",
            "aggregation": "per-prefix metrics per device and seed; mean seeds per device; equal-device mean",
            "source_prediction_manifest": "evaluation/prediction_inputs_manifest.json",
        })
    summary = pd.DataFrame(records)
    EVALUATION.mkdir(parents=True, exist_ok=True)
    superseded = EVALUATION / "superseded_v2_runner"
    superseded.mkdir(parents=True, exist_ok=True)
    superseded_records = []
    for dataset in batch["datasets"]:
        for name in ("test_predictions.csv", "validation_predictions.csv", "device_level_metrics.csv", "summary_metrics.csv"):
            source = OUTPUT / dataset / name
            if source.is_file():
                target = superseded / dataset / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(source, target)
                superseded_records.append({
                    "path": str(source.relative_to(OUTPUT)),
                    "archived_path": str(target.relative_to(OUTPUT)),
                    "sha256": sha256(target),
                    "status": "SUPERSEDED_WRONG_PARAMETERS",
                })
    archive_manifest = EVALUATION / "superseded_v2_runner_manifest.json"
    if not archive_manifest.exists():
        archive_manifest.write_text(json.dumps({
            "reason": "Prior v2_runner outputs were produced with incorrect experiment parameters.",
            "archived_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": superseded_records,
        }, indent=2) + "\n", encoding="utf-8")

    # Refresh the dataset-level outputs used by existing downstream tables.
    for dataset in batch["datasets"]:
        test = predictions[predictions.dataset.eq(dataset)].copy()
        validation = validations[validations.dataset.eq(dataset)].copy()
        test.to_csv(OUTPUT / dataset / "test_predictions.csv", index=False)
        validation.to_csv(OUTPUT / dataset / "validation_predictions.csv", index=False)
        dataset_units = per_device_seed[per_device_seed.dataset.eq(dataset)].copy()
        dataset_units.to_csv(OUTPUT / dataset / "device_level_metrics.csv", index=False)
        dataset_summary = summary[summary.dataset.eq(dataset)].copy()
        dataset_summary.to_csv(OUTPUT / dataset / "summary_metrics.csv", index=False)

    summary_path = EVALUATION / "metrics_summary.csv"
    summary.to_csv(summary_path, index=False)
    inputs_path = EVALUATION / "prediction_inputs_manifest.json"
    inputs_path.write_text(json.dumps({
        "protocol_id": batch["protocol_id"],
        "batch_run_id": batch["batch_run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_count": len(input_records),
        "predictions": input_records,
    }, indent=2) + "\n", encoding="utf-8")

    evaluation_manifest = {
        "record_type": "metrics_manifest",
        "protocol_id": batch["protocol_id"],
        "batch_run_id": batch["batch_run_id"],
        "evaluation_code_commit": batch.get("source_commit"),
        "evaluation_code_sha256": sha256(Path(__file__).resolve()),
        "prediction_inputs_manifest": "evaluation/prediction_inputs_manifest.json",
        "prediction_inputs_manifest_sha256": sha256(inputs_path),
        "metrics_output_path": "evaluation/metrics_summary.csv",
        "metrics_output_sha256": sha256(summary_path),
        "metrics": ["RMSE", "MAE", "P90AE"],
        "aggregation": "Compute metrics over prefixes per device per seed; average seeds within device; then take the equal-device mean.",
        "status": "COMPLETED",
    }
    manifest_path = EVALUATION / "metrics_manifest.json"
    manifest_path.write_text(json.dumps(evaluation_manifest, indent=2) + "\n", encoding="utf-8")
    batch["metrics_manifest_sha256"] = sha256(manifest_path)
    batch["metrics_output_sha256"] = sha256(summary_path)
    batch["prediction_inputs_manifest_sha256"] = sha256(inputs_path)
    batch["superseded_v2_runner_manifest"] = str(archive_manifest.relative_to(OUTPUT))
    batch["superseded_v2_runner_manifest_sha256"] = sha256(archive_manifest)
    batch["batch_status"] = "COMPLETED_WITH_METRICS"
    temporary = BATCH_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(batch, indent=2) + "\n", encoding="utf-8")
    temporary.replace(BATCH_PATH)


if __name__ == "__main__":
    main()
