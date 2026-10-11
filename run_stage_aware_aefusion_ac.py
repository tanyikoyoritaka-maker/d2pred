#!/usr/bin/env python3
"""Train and validate the theory-aligned Wiener/STL/AE fusion on AC data."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from analysis.causal_decomposition import (
    CausalDecompositionConfig,
    StableStageConfig,
    freeze_fusion_model,
    rolling_stl_trend,
    stable_stage_wiener_expected_trend,
)
from analysis.theory_aligned_decomposition import (
    StageAwareFusionConfig,
    StageAwareTrendAutoencoder,
    stage_aware_causal_decompose,
    stage_aware_fusion_loss,
    smooth_stl_candidate,
    wiener_reliability,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/stage_aware_aefusion_ac"
SEED = 42
DECOMPOSITION = CausalDecompositionConfig()
FUSION = StageAwareFusionConfig()
STABLE = {
    "ACCom": StableStageConfig(
        k_min=12, gamma_r=4.0, gamma_s=4.0, rolling_window=8, consecutive_q=3
    ),
    "ACSim": StableStageConfig(
        k_min=20, gamma_r=3.0, gamma_s=3.0, rolling_window=10, consecutive_q=2
    ),
}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def endpoints(length: int) -> list[int]:
    values = list(range(min(30, length), length + 1, 20))
    if not values or values[-1] != length:
        values.append(length)
    if len(values) > 20:
        indices = np.linspace(0, len(values) - 1, 20).round().astype(int)
        values = [values[index] for index in indices]
    return values


def fusion_item(
    di: np.ndarray,
    stable: StableStageConfig,
    fusion_config: StageAwareFusionConfig = FUSION,
) -> dict:
    wiener, metadata = stable_stage_wiener_expected_trend(
        di, stable, DECOMPOSITION
    )
    stl, _ = rolling_stl_trend(di, DECOMPOSITION)
    stl = smooth_stl_candidate(stl, fusion_config.stl_post_smoothing_fraction)
    return {
        "trends": np.stack((wiener, stl)).astype(np.float32),
        "di": np.asarray(di, dtype=np.float32),
        "stable_fraction": np.float32(metadata["stable_end"] / len(di)),
        "reliability": np.float32(wiener_reliability(di, wiener, stl, metadata)),
    }


class FusionDataset(Dataset):
    def __init__(self, values: list[dict]) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> dict:
        return self.values[index]


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    width = max(len(item["di"]) for item in batch)
    result = {
        "trends": torch.zeros(len(batch), 2, width),
        "di": torch.zeros(len(batch), width),
        "mask": torch.zeros(len(batch), width, dtype=torch.bool),
        "stable_fraction": torch.tensor([item["stable_fraction"] for item in batch]),
        "reliability": torch.tensor([item["reliability"] for item in batch]),
    }
    for index, item in enumerate(batch):
        length = len(item["di"])
        result["trends"][index, :, :length] = torch.from_numpy(item["trends"])
        result["di"][index, :length] = torch.from_numpy(item["di"])
        result["mask"][index, :length] = True
    return result


def prepare(
    frame: pd.DataFrame,
    split: str,
    stable: StableStageConfig,
    fusion_config: StageAwareFusionConfig = FUSION,
) -> list[dict]:
    result = []
    selected = frame[frame.split.eq(split)]
    for _, group in selected.groupby("unit_id", sort=True):
        di = group.sort_values("cycle").DI.to_numpy(np.float32)
        for end in endpoints(len(di)):
            result.append(fusion_item(di[:end], stable, fusion_config))
    return result


def train_fusion(
    frame: pd.DataFrame,
    stable: StableStageConfig,
    device: torch.device,
    epochs: int,
    fusion_config: StageAwareFusionConfig = FUSION,
    prepared: dict[str, list[dict]] | None = None,
) -> tuple[StageAwareTrendAutoencoder, list[dict], int]:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    prepared = prepared or {
        split: prepare(frame, split, stable, fusion_config) for split in ("train", "val")
    }
    loaders = {
        split: DataLoader(
            FusionDataset(prepared[split]),
            batch_size=16,
            shuffle=split == "train",
            collate_fn=collate,
        )
        for split in ("train", "val")
    }
    model = StageAwareTrendAutoencoder(config=fusion_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = (float("inf"), None, 0)
    logs = []
    for epoch in range(1, epochs + 1):
        epoch_values = {"train": [], "val": []}
        for role in ("train", "val"):
            model.train(role == "train")
            context = torch.enable_grad() if role == "train" else torch.no_grad()
            with context:
                for batch in loaders[role]:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    if role == "train":
                        optimizer.zero_grad(set_to_none=True)
                    output, details = model.forward_components(
                        batch["trends"], batch["mask"],
                        batch["stable_fraction"], batch["reliability"],
                    )
                    loss, components = stage_aware_fusion_loss(
                        output, batch["di"], batch["trends"], batch["mask"],
                        details, fusion_config,
                    )
                    if role == "train":
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                    valid = batch["mask"].sum().clamp_min(1)
                    epoch_values[role].append({
                        "loss": float(loss.item()),
                        **{key: float(value.item()) for key, value in components.items()},
                        "alpha_mean": float(details["alpha"].sum().item() / valid.item()),
                        "correction_rms": float(torch.sqrt(
                            (details["correction"] ** 2).sum() / valid
                        ).item()),
                    })
        row = {"epoch": epoch}
        for role, values in epoch_values.items():
            for key in values[0]:
                row[f"{role}_{key}"] = float(np.mean([value[key] for value in values]))
        logs.append(row)
        if row["val_loss"] < best[0] - 1e-8:
            best = (row["val_loss"], deepcopy(model.state_dict()), epoch)
    if best[1] is None:
        raise RuntimeError("No stage-aware fusion checkpoint was selected")
    model.load_state_dict(best[1])
    freeze_fusion_model(model)
    return model, logs, best[2]


def low_frequency_ratio(values: np.ndarray, window: int = 51) -> float:
    low = pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()
    return float(np.var(low) / max(float(np.var(values)), 1e-8))


def audit_dataset(
    name: str,
    frame: pd.DataFrame,
    stable: StableStageConfig,
    model: StageAwareTrendAutoencoder,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    rows = []
    curves = {}
    for unit, group in frame.groupby("unit_id", sort=True):
        group = group.sort_values("cycle")
        di = group.DI.to_numpy(np.float32)
        dwc, edc, metadata = stage_aware_causal_decompose(
            di, model, stable, DECOMPOSITION, device
        )
        di_variance = max(float(np.var(di)), 1e-8)
        dwc_rul = abs(float(np.corrcoef(dwc, group.RUL.to_numpy(float))[0, 1]))
        edc_rul = abs(float(np.corrcoef(edc, group.RUL.to_numpy(float))[0, 1]))
        rows.append({
            "dataset": name,
            "unit": int(unit),
            "split": str(group.split.iloc[0]),
            "length": len(di),
            "dwc_std_ratio": float(np.std(dwc) / max(float(np.std(di)), 1e-8)),
            "dwc_range_ratio": float(np.ptp(dwc) / max(float(np.ptp(di)), 1e-8)),
            "dwc_di_corr": float(np.corrcoef(dwc, di)[0, 1]),
            "edc_variance_ratio": float(np.var(edc) / di_variance),
            "edc_low_frequency_ratio": low_frequency_ratio(edc),
            "abs_dwc_rul_corr": dwc_rul,
            "abs_edc_rul_corr": edc_rul,
            "dwc_rul_dominates": dwc_rul > edc_rul,
            "stable_fraction": metadata["stable_fraction"],
            "wiener_reliability": metadata["wiener_reliability"],
            "alpha_mean": metadata["alpha_mean"],
            "correction_rms": metadata["correction_rms"],
            "identity_error": metadata["edc_identity_max_abs_error"],
        })
        curves[str(unit)] = {
            "cycle": group.cycle.to_numpy(float), "di": di, "dwc": dwc, "edc": edc,
            "split": str(group.split.iloc[0]),
        }
    return pd.DataFrame(rows), curves


def summarize(values: pd.DataFrame) -> dict:
    summary = {
        "num_units": int(len(values)),
        "median_dwc_std_ratio": float(values.dwc_std_ratio.median()),
        "median_dwc_range_ratio": float(values.dwc_range_ratio.median()),
        "median_dwc_di_corr": float(values.dwc_di_corr.median()),
        "median_edc_variance_ratio": float(values.edc_variance_ratio.median()),
        "median_edc_low_frequency_ratio": float(values.edc_low_frequency_ratio.median()),
        "median_abs_dwc_rul_corr": float(values.abs_dwc_rul_corr.median()),
        "median_abs_edc_rul_corr": float(values.abs_edc_rul_corr.median()),
        "dwc_rul_dominates_count": int(values.dwc_rul_dominates.sum()),
        "max_identity_error": float(values.identity_error.max()),
    }
    summary["physical_checks"] = {
        "median_dwc_std_ratio_between_0.5_and_1.25": 0.5 <= summary["median_dwc_std_ratio"] <= 1.25,
        "median_edc_variance_ratio_below_0.15": summary["median_edc_variance_ratio"] < 0.15,
        "median_edc_low_frequency_ratio_below_0.65": summary["median_edc_low_frequency_ratio"] < 0.65,
        "dwc_rul_dominates_at_least_75_percent": summary["dwc_rul_dominates_count"] >= int(np.ceil(0.75 * len(values))),
        "identity_exact": summary["max_identity_error"] <= 1e-6,
    }
    summary["passed"] = all(summary["physical_checks"].values())
    return summary


def plot_curves(all_curves: dict[str, dict[str, dict[str, np.ndarray]]], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
    for row, name in enumerate(("ACCom", "ACSim")):
        candidates = all_curves[name]
        test_units = [unit for unit, values in candidates.items() if values["split"] == "test"]
        unit = sorted(test_units, key=int)[0] if test_units else sorted(candidates, key=int)[0]
        values = candidates[unit]
        axes[row, 0].plot(values["cycle"], values["di"], label="Raw DI", color="#3b6fb6", lw=1.2)
        axes[row, 0].plot(values["cycle"], values["dwc"], label="DWC", color="#2a9d55", lw=2.0)
        axes[row, 0].set_title(f"{name} unit {unit}: DI and stage-aware DWC")
        axes[row, 0].set_ylabel("degradation index")
        axes[row, 0].legend()
        axes[row, 1].plot(values["cycle"], values["edc"], label="EDC = DI - DWC", color="#d1495b", lw=1.1)
        axes[row, 1].axhline(0.0, color="black", lw=0.7, alpha=0.6)
        axes[row, 1].set_title(f"{name} unit {unit}: event disturbance residual")
        axes[row, 1].legend()
        for axis in axes[row]:
            axis.set_xlabel("cycle")
            axis.grid(alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    all_audits = []
    all_curves = {}
    summaries = {}
    for name in ("ACCom", "ACSim"):
        data = pd.read_csv(ROOT / "data/industrial/DI_df.csv")
        groups = pd.read_csv(ROOT / "manifests/group_manifest.csv")
        frame = data.merge(groups[["unit", "group"]], on="unit")
        frame = frame[frame.group.eq(name)].rename(columns={"time_cycles":"cycle", "HI":"DI"})
        frame["unit_id"] = frame.unit.astype(str); frame["split"] = "train"; frame["subset_or_condition"] = name
        kind = "AC"
        if kind != "AC":
            raise AssertionError(f"Expected AC dataset, got {kind}")
        run_dir = OUT / "runs" / name
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{name}] preparing prefix-causal raw-scale Wiener/STL candidates", flush=True)
        model, logs, best_epoch = train_fusion(
            frame, STABLE[name], device, args.epochs
        )
        pd.DataFrame(logs).to_csv(run_dir / "fusion_training_log.csv", index=False)
        torch.save({
            "model_state_dict": model.state_dict(),
            "architecture": model.architecture_metadata(),
            "stable_stage": asdict(STABLE[name]),
            "decomposition": asdict(DECOMPOSITION),
            "best_epoch": best_epoch,
            "seed": SEED,
        }, run_dir / "fusion_checkpoint.pt")
        audit, curves = audit_dataset(name, frame, STABLE[name], model, device)
        audit.to_csv(run_dir / "full_device_decomposition_audit.csv", index=False)
        summary = summarize(audit)
        summary.update({"dataset": name, "best_epoch": best_epoch})
        write_json(run_dir / "decomposition_summary.json", summary)
        print(f"[{name}] physical summary: {summary}", flush=True)
        all_audits.append(audit)
        all_curves[name] = curves
        summaries[name] = summary
    pd.concat(all_audits, ignore_index=True).to_csv(
        OUT / "full_device_decomposition_audit.csv", index=False
    )
    write_json(OUT / "decomposition_summary.json", summaries)
    plot_curves(all_curves, OUT / "figures" / "stage_aware_di_dwc_edc.png")
    if not all(value["passed"] for value in summaries.values()):
        raise AssertionError("Stage-aware decomposition failed one or more physical checks")
    print(f"All physical checks passed. Figure: {OUT / 'figures' / 'stage_aware_di_dwc_edc.png'}")


if __name__ == "__main__":
    main()
