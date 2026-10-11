#!/usr/bin/env python3
"""Nested device-wise public-data rerun under the approved reviewer protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, ShuffleSplit

from run_ac_nested_gru_rerun import (
    apply_validation_calibration, checksum_manifest, now, paired_tests, reliability, summarize, train_one,
    unit_metrics, write_json,
)
import run_ac_nested_gru_rerun as ac_runner
from run_unified_theory_baseline_benchmark import (
    FUSION_CONFIG, TRANSFER_STABLE, add_decomposition, fusion_prepared,
    raw_prefix_rows,
)
from run_causal_stable_stage import load_spec
from run_stage_aware_aefusion_ac import train_fusion


ROOT = Path(__file__).resolve().parent


def folds_for(frame: pd.DataFrame, count: int) -> list[dict]:
    units = np.asarray(sorted(frame.unit_id.astype(str).unique()))
    outer = KFold(count, shuffle=True, random_state=20260915)
    result = []
    for fold, (development_indices, test_indices) in enumerate(outer.split(units)):
        development = units[development_indices]
        test = units[test_indices]
        validation_count = max(1, int(round(0.22 * len(development))))
        validation_count = min(validation_count, len(development) - 1)
        inner = ShuffleSplit(1, test_size=validation_count, random_state=7100 + fold)
        train_indices, validation_indices = next(inner.split(development))
        result.append({
            "outer_fold": fold,
            "train_units": sorted(development[train_indices].tolist()),
            "validation_units": sorted(development[validation_indices].tolist()),
            "test_units": sorted(test.tolist()),
        })
    return result


def assigned_frame(frame: pd.DataFrame, fold: dict) -> pd.DataFrame:
    roles = {}
    for role in ("train", "validation", "test"):
        roles.update({str(unit): role for unit in fold[f"{role}_units"]})
    output = frame.copy()
    output["unit_id"] = output.unit_id.astype(str)
    output["split"] = output.unit_id.map(roles).replace({"validation": "val"})
    if output.split.isna().any():
        raise AssertionError("Fold assignment does not cover all devices")
    return output


def prepare_fold(dataset: str, frame: pd.DataFrame, kind: str, fold: dict,
                 out: Path, device: torch.device, maximum_prefixes: int,
                 fusion_epochs: int, aep_gamma: float, aep_min_size: int,
                 aep_max_points: int, aep_max_tokens: int,
                 healthy_method: str, healthy_params: dict) -> list[dict]:
    cache = out / "cache" / dataset / f"outer_fold_{fold['outer_fold']}_rows.pt"
    if cache.exists():
        rows = torch.load(cache, map_location="cpu", weights_only=False)["rows"]
        for row in rows:
            row["token_mask"] = np.asarray(row["token_mask"], dtype=bool)
        return rows
    assigned = assigned_frame(frame, fold)
    prepared = {
        split: fusion_prepared(assigned, split, kind, TRANSFER_STABLE, maximum_prefixes)
        for split in ("train", "val")
    }
    fusion, logs, best_epoch = train_fusion(
        assigned, TRANSFER_STABLE, device, fusion_epochs,
        fusion_config=FUSION_CONFIG, prepared=prepared,
    )
    fusion_dir = out / "fusion" / dataset / f"outer_fold_{fold['outer_fold']}"
    fusion_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(logs).to_csv(fusion_dir / "training_log.csv", index=False)
    torch.save({
        "model_state_dict": fusion.state_dict(), "best_epoch": best_epoch,
        "train_units": fold["train_units"],
        "validation_units": fold["validation_units"], "test_used": False,
    }, fusion_dir / "fusion_checkpoint.pt")
    base = raw_prefix_rows(dataset, assigned, kind, maximum_prefixes)
    decomposed, diagnostics = add_decomposition(
        base, fusion, TRANSFER_STABLE, device, aep_gamma, aep_min_size,
        aep_max_points, aep_max_tokens, healthy_method, healthy_params,
    )
    rows = []
    for row in decomposed:
        rows.append({
            "unit": row["unit"], "group": dataset,
            "split": "validation" if row["split"] == "val" else row["split"],
            "endpoint": row["endpoint"], "prefix_length": row["prefix_length"],
            "true_rul_cycles": row["true_output"],
            "di": row["raw_di"], "dwc": row["dwc"], "edc": row["edc"],
            "trend_candidates": row["trend_candidates"],
            "stable_fraction": row["stable_fraction"],
            "wiener_reliability": row["wiener_reliability"],
            "tokens": row["aep_tokens"],
            "token_mask": np.asarray(row["aep_token_mask"], dtype=bool),
        })
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rows": rows, "fold": fold}, cache)
    pd.DataFrame(diagnostics).to_csv(
        out / "cache" / dataset / f"outer_fold_{fold['outer_fold']}_diagnostics.csv",
        index=False,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="xjtu_sy,pronostia")
    parser.add_argument("--models", default="gru_d2pred_v2,d2pred_expert_exact_v1")
    parser.add_argument("--seeds", default="17,42,73")
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--fusion-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--maximum-prefixes", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aep-gamma", type=float, default=2.5)
    parser.add_argument("--aep-min-size", type=int, default=8)
    parser.add_argument("--aep-max-points", type=int, default=15)
    parser.add_argument("--aep-max-tokens", type=int, default=16)
    parser.add_argument("--healthy-method", choices=("current", "increment_rate"), default="current")
    parser.add_argument("--healthy-initial-window", type=int, default=20)
    parser.add_argument("--healthy-rate-mad-multiplier", type=float, default=3.0)
    parser.add_argument("--healthy-min-rate-sign-consistency", type=float, default=0.8)
    parser.add_argument("--healthy-consecutive-q", type=int, default=2)
    args = parser.parse_args()
    ac_runner.SENSITIVITY.update({"gamma": args.aep_gamma, "min_size": args.aep_min_size,
                                  "max_points": args.aep_max_points, "max_tokens": args.aep_max_tokens})
    healthy_params = {
        "initial_window": args.healthy_initial_window,
        "rate_mad_multiplier": args.healthy_rate_mad_multiplier,
        "consecutive_violations": args.healthy_consecutive_q,
        "minimum_rate_sign_consistency": args.healthy_min_rate_sign_consistency,
    }
    datasets = tuple(x.strip() for x in args.datasets.split(",") if x.strip())
    models = tuple(x.strip() for x in args.models.split(",") if x.strip())
    seeds = tuple(int(x) for x in args.seeds.split(","))
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    protocol = vars(args).copy(); protocol.update({
        "datasets": datasets, "models": models, "seeds": seeds,
        "split": "nested physical-device 3-fold",
        "normalization": "inner-training devices only",
        "test_used_for_selection": False,
    })
    protocol["output"] = str(protocol["output"]); write_json(out / "protocol.json", protocol)
    all_validation, all_test, all_cost, fold_manifest = [], [], [], {}
    for dataset in datasets:
        frame, _, kind = load_spec(dataset)
        folds = folds_for(frame, args.outer_folds); fold_manifest[dataset] = folds
        for fold in folds:
            print(f"[{dataset}][fold {fold['outer_fold']}] causal decomposition", flush=True)
            rows = prepare_fold(dataset, frame, kind, fold, out, device,
                                args.maximum_prefixes, args.fusion_epochs,
                                args.aep_gamma, args.aep_min_size,
                                args.aep_max_points, args.aep_max_tokens,
                                args.healthy_method, healthy_params)
            for seed in seeds:
                for model in models:
                    print(f"[{dataset}][fold {fold['outer_fold']}][seed {seed}][{model}]", flush=True)
                    validation, test, metadata = train_one(
                        rows, model, fold["outer_fold"], seed,
                        out / dataset, device, args.epochs, args.batch_size,
                        fusion_checkpoint_path=(
                            out / "fusion" / dataset / f"outer_fold_{fold['outer_fold']}" / "fusion_checkpoint.pt"
                        ),
                    )
                    all_validation.extend(validation); all_test.extend(test)
                    all_cost.append(metadata | {"dataset": dataset})
    write_json(out / "folds.json", fold_manifest)
    validation = pd.DataFrame(all_validation); test = pd.DataFrame(all_test)
    validation, test, calibration = apply_validation_calibration(validation, test)
    pd.DataFrame(calibration).to_csv(out / "validation_calibration.csv", index=False)
    validation.to_csv(out / "validation_predictions.csv", index=False)
    test.to_csv(out / "test_predictions.csv", index=False)
    unit = unit_metrics(test); unit.to_csv(out / "device_level_metrics.csv", index=False)
    summary = summarize(unit); summary.to_csv(out / "summary_metrics.csv", index=False)
    paired_tests(unit).to_csv(out / "paired_statistical_tests.csv", index=False)
    ensemble, reliability_summary = reliability(validation, test)
    ensemble.to_csv(out / "seed_ensemble_predictions_with_intervals.csv", index=False)
    reliability_summary.to_csv(out / "reliability_summary.csv", index=False)
    pd.DataFrame(all_cost).to_csv(out / "computational_cost.csv", index=False)
    write_json(out / "run_status.json", {"status": "verified_complete", "completed": now(),
                                         "num_test_predictions": len(test)})
    checksum_manifest(out); print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
