#!/usr/bin/env python3
"""Train one approved strict ablation after the protocol is validated."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.strict_ablation import ControlledExpertExactD2Pred, resolve_variant
from analysis.theory_aligned_decomposition import stage_aware_fusion_loss
from run_ac_nested_gru_rerun import (
    PrefixDataset,
    fit_normalization,
    load_ac,
    predict,
    seed_everything,
    summarize,
    unit_metrics,
    validate_predefined_folds,
    write_json,
)
from ablation_final.strict_pipeline import build_variant_fold_rows, sha256


PROTOCOL = ROOT / "ablation_final/protocol/ablation_manifest_v2.json"
DEFAULT_OUTPUT = ROOT / "ablation_final/outputs/strict_ablation_v2"
SEEDS = (17, 42, 73)


def train_one_variant(
    rows: list[dict],
    dataset: str,
    variant: str,
    fold: int,
    seed: int,
    output_root: Path,
    fusion_checkpoint: Path | None,
    device: torch.device,
    epochs: int,
    batch_size: int,
):
    spec = resolve_variant(variant)
    seed_everything(seed)
    split_rows = {
        role: [row for row in rows if row["split"] == role]
        for role in ("train", "validation", "test")
    }
    train_units = sorted({row["unit"] for row in split_rows["train"]})
    target_scale = max(row["true_rul_cycles"] for row in split_rows["train"]) + 1.0
    normalization = fit_normalization(rows)
    loaders = {
        role: DataLoader(
            PrefixDataset(values, normalization, target_scale, f"d2pred_expert_ablation_{spec.variant}"),
            batch_size=batch_size,
            shuffle=role == "train",
            num_workers=0,
        )
        for role, values in split_rows.items()
    }
    model = ControlledExpertExactD2Pred(spec.variant).to(device)
    if model.core.trend_fusion is not None:
        if fusion_checkpoint is None or not fusion_checkpoint.is_file():
            raise FileNotFoundError(f"Missing variant-specific fold fusion checkpoint: {fusion_checkpoint}")
        checkpoint = torch.load(fusion_checkpoint, map_location=device, weights_only=False)
        model.core.trend_fusion.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_loss, best_state, best_epoch = float("inf"), None, 0
    patience = 0
    history = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        if epoch == 26:
            for group in optimizer.param_groups:
                group["lr"] = 5e-4
        epoch_losses = {}
        for role in ("train", "validation"):
            model.train(role == "train")
            values = []
            context = torch.enable_grad() if role == "train" else torch.no_grad()
            with context:
                for inputs, labels, _ in loaders[role]:
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                    labels = labels.to(device)
                    if role == "train":
                        optimizer.zero_grad(set_to_none=True)
                    details = model(inputs, return_details=True)
                    loss = criterion(details["prediction"], labels)
                    if spec.gradual_loss_active and details["gradual_prediction"] is not None:
                        loss = loss + 0.1 * criterion(details["gradual_prediction"], labels)
                    if spec.event_loss_active and details["event_prediction"] is not None:
                        loss = loss + 0.1 * criterion(details["event_prediction"], labels)
                    if spec.trend_loss_active:
                        if details["trend_fusion_details"] is None:
                            raise RuntimeError(f"{spec.variant} requires fold trend-fusion details")
                        fusion_loss, _ = stage_aware_fusion_loss(
                            details["fused_dwc"], inputs["di"], inputs["trend_candidates"],
                            inputs["sequence_mask"], details["trend_fusion_details"],
                            model.core.trend_fusion.config,
                        )
                        loss = loss + 0.1 * fusion_loss
                    if role == "train":
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"Nonfinite objective: {spec.variant}, fold={fold}, seed={seed}")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                    values.append(float(loss.detach().item()))
            epoch_losses[role] = float(np.mean(values))
        history.append({
            "epoch": epoch,
            "train_objective": epoch_losses["train"],
            "validation_objective": epoch_losses["validation"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        if epoch_losses["validation"] < best_loss - 1e-8:
            best_loss = epoch_losses["validation"]
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0
        elif epoch > 25:
            patience += 1
        else:
            patience = 0
        if epoch > 25 and patience >= 8:
            break

    if best_state is None:
        raise RuntimeError("No validation-selected checkpoint was produced")
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    model_name = f"d2pred_expert_ablation_{spec.variant}"
    run_dir = output_root / dataset / spec.variant / f"outer_fold_{fold}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(run_dir / "training_log.csv", index=False)
    metadata = {
        "dataset": dataset,
        "variant": spec.variant,
        "outer_fold": fold,
        "seed": seed,
        "parent_architecture": "d2pred_expert_exact_v1",
        "architecture": model.architecture_metadata(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "target_scale_cycles": target_scale,
        "target_scale_fit_units": train_units,
        "normalization_fit_units": normalization["fit_units"],
        "fusion_checkpoint": str(fusion_checkpoint) if fusion_checkpoint else None,
        "best_epoch": best_epoch,
        "best_validation_objective": best_loss,
        "training_seconds": elapsed,
        "test_used_for_selection": False,
    }
    torch.save({"model_state_dict": best_state, **metadata}, run_dir / "best_checkpoint.pt")
    write_json(run_dir / "metadata.json", metadata)
    validation = predict(
        model, model_name, loaders["validation"], split_rows["validation"],
        target_scale, device, fold, seed, "validation",
    )
    test = predict(
        model, model_name, loaders["test"], split_rows["test"],
        target_scale, device, fold, seed, "test",
    )
    for record in validation + test:
        record["model"] = spec.variant
    return validation, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Required execution confirmation")
    parser.add_argument("--dataset", choices=("ACSim", "ACCom", "ACAll"), required=True)
    parser.add_argument("--variant", choices=tuple(VARIANT for VARIANT in (
        "no_wiener", "no_stl", "no_autoencoder_fusion", "no_dwc", "no_edc",
        "no_decomposition", "no_fusion", "no_aep",
    )), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--fusion-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fold", type=int, choices=(0, 1, 2))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if not args.execute:
        parser.error("Training is disabled unless --execute is specified after separate review")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "VERIFIED_COMPLETED" or protocol.get("execution_enabled") is not True:
        raise RuntimeError("Strict ablation protocol is not approved for execution")

    dataset, variant = args.dataset, resolve_variant(args.variant).variant
    expected_output = DEFAULT_OUTPUT.resolve()
    if args.output_root.resolve() != expected_output:
        raise RuntimeError(f"Output must be isolated at {expected_output}")
    if dataset not in protocol["data"]["datasets"] or variant not in protocol["variants"]:
        raise RuntimeError(f"Unregistered dataset/variant in candidate protocol: {dataset}/{variant}")
    source_path = ROOT / "data/industrial/DI_df.csv"
    expected_source = Path(protocol["data"]["path"])
    if expected_source.resolve() != source_path.resolve():
        raise RuntimeError(f"Protocol data source does not match the canonical DI source: {expected_source}")
    if sha256(source_path) != protocol["data"]["sha256"]:
        raise RuntimeError("Canonical DI source hash differs from the candidate protocol")

    data = load_ac()
    if dataset == "ACAll":
        frame = data.copy()
        frame["group"] = "ACAll"
    else:
        frame = data[data.group.eq(dataset)].copy()
    folds_path = ROOT / f"results/frozen_protocol_final/comparison/{dataset}/folds.json"
    expected_fold = protocol["data"]["fold_manifests"][dataset]
    if folds_path.resolve() != Path(expected_fold["path"]).resolve():
        raise RuntimeError("Fold manifest path differs from the candidate protocol")
    if sha256(folds_path) != expected_fold["sha256"]:
        raise RuntimeError("Fold manifest hash differs from the candidate protocol")
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    folds = validate_predefined_folds(frame, folds)
    device = torch.device(args.device)
    predictions = {"validation": [], "test": []}
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / dataset / variant / "run_status.json", {
        "status": "running", "protocol": str(PROTOCOL), "dataset": dataset,
        "variant": variant, "started_at": datetime.now(timezone.utc).isoformat(),
    })
    try:
        for fold in folds:
            fold_id = int(fold["outer_fold"])
            if args.fold is not None and fold_id != args.fold:
                continue
            rows, fusion_checkpoint, cache = build_variant_fold_rows(
                frame, fold, dataset, variant, args.output_root, device,
                fusion_epochs=args.fusion_epochs,
            )
            for seed in (args.seed,) if args.seed is not None else SEEDS:
                validation, test = train_one_variant(
                    rows, dataset, variant, fold_id, seed,
                    args.output_root, fusion_checkpoint, device,
                    args.epochs, args.batch_size,
                )
                run_dir = args.output_root / dataset / variant / f"outer_fold_{fold_id}" / f"seed_{seed}"
                pd.DataFrame(validation).to_csv(run_dir / "validation_predictions.csv", index=False)
                pd.DataFrame(test).to_csv(run_dir / "test_predictions.csv", index=False)
                predictions["validation"].extend(validation)
                predictions["test"].extend(test)
            write_json(args.output_root / dataset / variant / "last_cache.json", {"path": str(cache)})
        root = args.output_root / dataset / variant
        if args.fold is None and args.seed is None:
            validation = pd.DataFrame(predictions["validation"])
            test = pd.DataFrame(predictions["test"])
            validation.to_csv(root / "validation_predictions.csv", index=False)
            test.to_csv(root / "test_predictions.csv", index=False)
            unit = unit_metrics(test)
            unit.to_csv(root / "device_level_metrics.csv", index=False)
            summarize(unit).to_csv(root / "summary_metrics.csv", index=False)
        write_json(root / "run_status.json", {
            "status": "run_completed" if args.fold is not None or args.seed is not None else "completed",
            "datasets": [dataset], "variant": variant,
            "outer_fold": args.fold, "seed": args.seed,
        })
    except Exception as exc:
        write_json(args.output_root / dataset / variant / "run_status.json", {
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
        })
        raise


if __name__ == "__main__":
    from analysis.strict_ablation import VARIANT_SPECS
    main()
