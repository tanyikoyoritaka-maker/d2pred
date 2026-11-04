#!/usr/bin/env python3
"""Leakage-safe nested device-wise AC rerun for the frozen GRU D2Pred model.

The historical 5.75/5.13 results are retained only as a reference.  This
runner removes device-specific lifetime normalization: every predictor target
scale is fitted from the inner-training devices of its outer fold.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from analysis.ac_cpm_core import causal_change_points, pad_tokens, segment_tokens
from analysis.frozen_gru_d2pred import FrozenGRUD2Pred
from analysis.expert_exact_d2pred import ExpertExactD2Pred
from analysis.theory_aligned_decomposition import (
    stage_aware_causal_decompose, stage_aware_fusion_loss,
)
from models.benchmark import build_model as build_baseline
from run_stage_aware_aefusion_ac import (
    DECOMPOSITION, STABLE, endpoints as _default_endpoints, train_fusion,
)


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT
DEFAULT_OUT = ROOT / "results/ac_nested_gru_final"
EXCLUDED_UNITS = {1, 3, 11, 35, 38}
SEEDS = (17, 42, 73)


def endpoints(length: int, start: int = 30, stride: int = 20,
              maximum: int = 20) -> list[int]:
    """Generate the registered causal prefix grid with explicit overrides."""
    start = max(1, int(start)); stride = max(1, int(stride)); maximum = max(1, int(maximum))
    values = list(range(min(start, length), length + 1, stride))
    if not values or values[-1] != length:
        values.append(length)
    if len(values) > maximum:
        indices = np.linspace(0, len(values) - 1, maximum).round().astype(int)
        values = [values[index] for index in indices]
    return values
CORE_MODELS = (
    "gru_d2pred_v2", "d2pred_expert_exact_v1", "d2pred_expert_age_v1",
    "d2pred_expert_cycle_v1", "d2pred_expert_prefix_v1",
    "d2pred_expert_age_calibrated_v1", "d2pred_expert_age_gate_v1",
    "d2pred_expert_age_gate_late_v1", "d2pred_expert_age_gate_late_mono_v1",
    "CNN", "Transformer",
    "TCN_BiGRU", "KGHM",
)
FULL_BASELINES = (
    "CNN", "DAG_network", "PM2FN",
    "Transformer", "TCN_BiGRU", "KGHM", "ATCN", "MDFA",
    "SBi_Transformer", "Attn_ResDSCNN",
)
FULL_MODELS = ("gru_d2pred_v2", "d2pred_expert_exact_v1") + FULL_BASELINES + CORE_MODELS[7:]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_ac(drop_units: set[int] | None = None, merge_units_to_accom: set[int] | None = None,
            group_manifest: Path | None = None, keep_standard_exclusions: bool = True) -> pd.DataFrame:
    data = pd.read_csv(ROOT / "data/industrial/DI_df.csv")
    groups = pd.read_csv(group_manifest or (ROOT / "manifests/group_manifest.csv"))
    if "usable" not in groups:
        groups["usable"] = True
    groups = groups[["unit", "group", "lifetime", "usable"]].copy()
    groups["unit"] = groups.unit.astype(int)
    data["unit"] = data.unit.astype(int)
    frame = data.merge(groups, on="unit", how="inner")
    if keep_standard_exclusions:
        frame = frame[~frame.unit.isin(EXCLUDED_UNITS)].copy()
    frame = frame[frame.usable.astype(bool)].copy()
    if drop_units:
        frame = frame[~frame.unit.isin(drop_units)].copy()
    if merge_units_to_accom:
        frame.loc[frame.unit.isin(merge_units_to_accom), "group"] = "ACCom"
    frame = frame.rename(columns={"time_cycles": "cycle", "HI": "DI"})
    frame["RUL"] = frame.RUL.astype(float)
    frame["DI"] = frame.DI.astype(float)
    return frame[["unit", "cycle", "RUL", "DI", "group", "lifetime"]]


def make_folds(frame: pd.DataFrame, outer_folds: int) -> list[dict]:
    devices = (
        frame.groupby("unit", as_index=False)
        .agg(group=("group", "first"), lifetime=("lifetime", "first"))
        .sort_values("unit")
        .reset_index(drop=True)
    )
    splitter = StratifiedKFold(outer_folds, shuffle=True, random_state=20260915)
    folds = []
    for fold, (development_index, test_index) in enumerate(
        splitter.split(devices.unit, devices.group)
    ):
        development = devices.iloc[development_index].reset_index(drop=True)
        test = devices.iloc[test_index]
        inner = StratifiedShuffleSplit(n_splits=1, test_size=0.22, random_state=7100 + fold)
        train_index, validation_index = next(inner.split(development.unit, development.group))
        train = development.iloc[train_index]
        validation = development.iloc[validation_index]
        item = {
            "outer_fold": fold,
            "train_units": sorted(train.unit.astype(int).tolist()),
            "validation_units": sorted(validation.unit.astype(int).tolist()),
            "test_units": sorted(test.unit.astype(int).tolist()),
        }
        all_sets = [set(item[key]) for key in ("train_units", "validation_units", "test_units")]
        if all_sets[0] & all_sets[1] or all_sets[0] & all_sets[2] or all_sets[1] & all_sets[2]:
            raise AssertionError("Device overlap in generated fold")
        item["group_counts"] = {
            role: devices[devices.unit.isin(item[f"{role}_units"])].group.value_counts().to_dict()
            for role in ("train", "validation", "test")
        }
        folds.append(item)
    return folds


def validate_predefined_folds(frame: pd.DataFrame, folds: list[dict]) -> list[dict]:
    """Validate reviewer-visible device folds before any model fitting."""
    expected = set(frame.unit.astype(int).unique())
    test_coverage = []
    for index, fold in enumerate(folds):
        if int(fold.get("outer_fold", -1)) != index:
            raise ValueError("Predefined folds must have consecutive outer_fold identifiers")
        roles = {role: set(map(int, fold[f"{role}_units"]))
                 for role in ("train", "validation", "test")}
        if roles["train"] & roles["validation"] or roles["train"] & roles["test"] or roles["validation"] & roles["test"]:
            raise ValueError(f"Device overlap in predefined fold {index}")
        if set.union(*roles.values()) != expected:
            raise ValueError(f"Predefined fold {index} does not cover retained devices exactly")
        for role, selected in roles.items():
            groups = set(frame.loc[frame.unit.isin(selected), "group"])
            if groups != set(frame.group.unique()):
                raise ValueError(f"Predefined fold {index} {role} does not contain every group")
        test_coverage.extend(roles["test"])
    if len(test_coverage) != len(expected) or set(test_coverage) != expected:
        raise ValueError("Every retained device must occur in exactly one predefined test fold")
    return folds


def split_frame(frame: pd.DataFrame, fold: dict) -> pd.DataFrame:
    role = {}
    for name in ("train", "validation", "test"):
        role.update({unit: name for unit in fold[f"{name}_units"]})
    result = frame.copy()
    result["split"] = result.unit.map(role)
    if result.split.isna().any():
        raise AssertionError("Fold does not cover every retained device")
    return result


def resize(values: np.ndarray, length: int = 128) -> np.ndarray:
    values = np.asarray(values, np.float32)
    if len(values) == 1:
        return np.full(length, values[0], np.float32)
    return np.interp(
        np.linspace(0, len(values) - 1, length), np.arange(len(values)), values
    ).astype(np.float32)


def event_tokens(edc: np.ndarray, *, sensitivity: float = 2.5,
                 min_size: int = 8, max_points: int = 15,
                 max_tokens: int = 16) -> tuple[np.ndarray, np.ndarray, list[int]]:
    points = causal_change_points(edc, sensitivity=sensitivity,
                                  min_size=min_size, max_points=max_points)
    token_values = segment_tokens(edc, points, mode="stats")
    tokens, mask = pad_tokens(token_values, max_tokens)
    return tokens.astype(np.float32), mask.astype(bool), points


def build_fold_rows(
    frame: pd.DataFrame,
    fold: dict,
    out: Path,
    device: torch.device,
    fusion_epochs: int,
    mix_groups: bool = False,
    aep_config: dict | None = None,
    prefix_config: dict | None = None,
) -> list[dict]:
    cache = out / "cache" / f"outer_fold_{fold['outer_fold']}_rows.pt"
    cache_signature = {"aep": aep_config or {"sensitivity": 2.5,
                                             "min_size": 8,
                                             "max_points": 15,
                                             "max_tokens": 16},
                      "prefix": prefix_config or {"start": 30, "stride": 20,
                                                   "maximum": 20},
                      "mix_groups": mix_groups,
                      "shared_decomposition": False}
    if cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        expected_cache_version = 2
        cache_matches = payload.get("cache_signature") == cache_signature
        # The frozen ACAll primary cache predates cache signatures.  It is
        # reusable only for the registered default AEP/prefix configuration.
        legacy_default = (payload.get("cache_signature") is None
                          and cache_signature == {"aep": {"sensitivity": 2.5,
                              "min_size": 8, "max_points": 15, "max_tokens": 16},
                              "prefix": {"start": 30, "stride": 20, "maximum": 20},
                              "mix_groups": True, "shared_decomposition": False})
        fusion_checkpoint = (out / "fusion" / f"outer_fold_{fold['outer_fold']}"
                             / "ACAll" / "fusion_checkpoint.pt")
        # A prefix cache is not sufficient to resume a fold: the fold-specific
        # trend-fusion checkpoint must also exist in this output package.
        # This prevents copied primary caches from bypassing fusion fitting.
        if (payload.get("cache_version") == expected_cache_version
                and (cache_matches or legacy_default)
                and fusion_checkpoint.exists()):
            return payload["rows"]
    assigned = split_frame(frame, fold)
    models = {}
    if mix_groups:
        group_names = ("ACAll",)
    elif frame.group.nunique() == 1:
        # A single-group run (e.g. --only-group ACSim/ACCom) must not try to
        # fit an empty fusion dataset for the other cohort.
        group_names = (str(frame.group.iloc[0]),)
    else:
        group_names = ("ACCom", "ACSim")
    stable_config = {"ACAll": STABLE["ACCom"], **STABLE}
    for group in group_names:
        group_frame = assigned.copy() if group == "ACAll" else assigned[assigned.group.eq(group)].copy()
        group_frame["split"] = group_frame.split.replace({"validation": "val"})
        group_frame = group_frame.rename(columns={"unit": "unit_id"})
        model, logs, best_epoch = train_fusion(
            group_frame, stable_config[group], device, fusion_epochs,
        )
        models[group] = model
        run_dir = out / "fusion" / f"outer_fold_{fold['outer_fold']}" / group
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(logs).to_csv(run_dir / "training_log.csv", index=False)
        torch.save({
            "model_state_dict": model.state_dict(),
            "architecture": model.architecture_metadata(),
            "stable_stage": asdict(stable_config[group]),
            "decomposition": asdict(DECOMPOSITION),
            "fit_units": sorted(group_frame[group_frame.split.eq("train")].unit_id.unique().tolist()),
            "validation_units": sorted(group_frame[group_frame.split.eq("val")].unit_id.unique().tolist()),
            "test_units_used": False,
            "best_epoch": best_epoch,
        }, run_dir / "fusion_checkpoint.pt")
    rows = []
    for unit, unit_frame in assigned.groupby("unit", sort=True):
        unit_frame = unit_frame.sort_values("cycle")
        di_full = unit_frame.DI.to_numpy(np.float32)
        group = str(unit_frame.group.iloc[0])
        role = str(unit_frame.split.iloc[0])
        for end in endpoints(len(unit_frame), **(prefix_config or {})):
            observed = unit_frame.iloc[:end]
            di = di_full[:end]
            decomposition_group = group
            dwc, edc, metadata = stage_aware_causal_decompose(
                di, models[decomposition_group], stable_config[decomposition_group],
                DECOMPOSITION, device, return_candidates=True,
            )
            tokens, mask, points = event_tokens(edc, **(aep_config or {}))
            rows.append({
                "unit": int(unit), "group": group, "split": role,
                "endpoint": float(observed.cycle.iloc[-1]), "prefix_length": int(end),
                "true_rul_cycles": float(observed.RUL.iloc[-1]),
                "di": resize(di), "dwc": resize(dwc), "edc": resize(edc),
                "trend_candidates": np.stack((
                    resize(metadata["wiener_candidate"]),
                    resize(metadata["stl_candidate"]),
                )).astype(np.float32),
                "stable_fraction": np.float32(metadata["stable_fraction"]),
                "wiener_reliability": np.float32(metadata["wiener_reliability"]),
                "tokens": tokens, "token_mask": mask,
                "change_points": points,
                "edc_identity_error": float(metadata["edc_identity_max_abs_error"]),
            })
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cache_version": 2,
                "cache_signature": cache_signature,
                "fold": fold, "rows": rows}, cache)
    return rows


def variant_rows(rows: list[dict], model_name: str) -> list[dict]:
    if model_name.startswith(("ablation_", "controlled_ablation_")):
        raise ValueError("Ablation aliases are disabled pending historical implementation review")
    return [dict(source) for source in rows]


def fit_normalization(rows: list[dict]) -> dict:
    train = [row for row in rows if row["split"] == "train"]
    result = {}
    for key in ("di", "dwc", "edc"):
        values = np.concatenate([row[key] for row in train]).astype(np.float64)
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_std"] = max(float(values.std()), 1e-6)
    valid = np.concatenate([
        row["tokens"][row["token_mask"]] for row in train
    ]).astype(np.float64)
    if valid.size:
        result["token_mean"] = valid.mean(axis=0).tolist()
        std = valid.std(axis=0)
        std[std < 1e-6] = 1.0
        result["token_std"] = std.tolist()
    else:
        width = int(train[0]["tokens"].shape[-1])
        result["token_mean"] = np.zeros(width, dtype=float).tolist()
        result["token_std"] = np.ones(width, dtype=float).tolist()
    result["fit_units"] = sorted({row["unit"] for row in train})
    return result


class PrefixDataset(Dataset):
    def __init__(self, rows: list[dict], normalization: dict, target_scale: float, model_name: str = "") -> None:
        self.rows = rows
        self.normalization = normalization
        self.target_scale = target_scale
        self.token_mean = np.asarray(normalization["token_mean"], np.float32)
        self.token_std = np.asarray(normalization["token_std"], np.float32)
        self.model_name = model_name

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        inputs = {}
        for key in ("di", "dwc", "edc"):
            inputs[key] = torch.from_numpy(((
                row[key] - self.normalization[f"{key}_mean"]
            ) / self.normalization[f"{key}_std"]).astype(np.float32))
        valid = row["token_mask"].astype(bool)
        tokens = np.zeros_like(row["tokens"], dtype=np.float32)
        tokens[valid] = (row["tokens"][valid] - self.token_mean) / self.token_std
        inputs["tokens"] = torch.from_numpy(tokens)
        inputs["token_mask"] = torch.from_numpy(valid)
        age_values = [
            float(row["endpoint"]) / max(self.target_scale, 1.0),
            float(row["prefix_length"]) / 128.0,
        ]
        if self.model_name == "d2pred_expert_cycle_v1":
            age_values = age_values[:1]
        elif self.model_name == "d2pred_expert_prefix_v1":
            age_values = age_values[1:]
        inputs["age"] = torch.tensor(age_values, dtype=torch.float32)
        candidates = (row["trend_candidates"] - self.normalization["di_mean"]) / self.normalization["di_std"]
        inputs["trend_candidates"] = torch.from_numpy(candidates.astype(np.float32))
        inputs["sequence_mask"] = torch.ones(inputs["di"].shape[0], dtype=torch.bool)
        inputs["stable_fraction"] = torch.tensor(row["stable_fraction"], dtype=torch.float32)
        inputs["wiener_reliability"] = torch.tensor(row["wiener_reliability"], dtype=torch.float32)
        label = np.float32(row["true_rul_cycles"] / self.target_scale)
        return inputs, torch.tensor(label), index


class RawDISingleChannel(nn.Module):
    """Strict no-decomposition baseline: one direct raw-DI vector channel.

    This model deliberately has no DWC encoder, EDC encoder, decomposition,
    token branch, or recurrent substitute.  The fixed-length causal DI prefix
    is normalized by PrefixDataset and mapped directly to the RUL output.
    """

    def __init__(self, width: int = 128, hidden: int = 64) -> None:
        super().__init__()
        self.regression = nn.Sequential(
            nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, inputs: dict[str, torch.Tensor], return_details: bool = False):
        prediction = self.regression(inputs["di"]).squeeze(-1)
        if return_details:
            return {"prediction": prediction}
        return prediction

    def architecture_metadata(self) -> dict:
        return {
            "architecture_version": "raw_di_single_channel_v1",
            "input": "causal raw DI prefix only",
            "dwc_encoder": None, "edc_encoder": None, "event_encoder": None,
            "decomposition": False,
            "output": "sigmoid_fraction_of_train_only_target_scale",
        }


def make_model(name: str) -> nn.Module:
    if name == "gru_d2pred_v2":
        return FrozenGRUD2Pred()
    if name == "d2pred_expert_exact_v1":
        return ExpertExactD2Pred()
    if name in {
        "d2pred_expert_age_v1", "d2pred_expert_age_calibrated_v1",
        "d2pred_expert_age_gate_v1", "d2pred_expert_age_gate_late_v1",
        "d2pred_expert_age_gate_late_mono_v1",
        "d2pred_expert_cycle_v1", "d2pred_expert_prefix_v1",
    }:
        age_dimension = 1 if name in {"d2pred_expert_cycle_v1", "d2pred_expert_prefix_v1"} else 2
        return ExpertExactD2Pred(
            use_age=True,
            adaptive_gate=name in {
                "d2pred_expert_age_gate_v1",
                "d2pred_expert_age_gate_late_v1",
                "d2pred_expert_age_gate_late_mono_v1",
            }, age_dimension=age_dimension,
        )
    if name.startswith(("ablation_", "controlled_ablation_")):
        raise ValueError("Ablation aliases are disabled pending historical implementation review")
    return build_baseline(name)


def forward_model(model: nn.Module, name: str, inputs: dict[str, torch.Tensor]):
    if name.startswith(("gru_", "d2pred_", "ablation_")):
        return model(inputs)
    if name == "KGHM":
        return model(inputs["di"])
    return model(inputs["di"])


def is_expert(name: str) -> bool:
    return name.startswith("d2pred_expert_")


def is_age_expert(name: str) -> bool:
    return is_expert(name) and name != "d2pred_expert_exact_v1"


def uses_late_loss(name: str) -> bool:
    return name in {"d2pred_expert_age_gate_late_v1", "d2pred_expert_age_gate_late_mono_v1"}


def uses_mono_loss(name: str) -> bool:
    return name == "d2pred_expert_age_gate_late_mono_v1"


def monotonic_pair_loss(predictions: torch.Tensor, indices: torch.Tensor,
                        rows: list[dict]) -> torch.Tensor:
    penalties = []
    by_unit: dict[object, list[tuple[float, int]]] = {}
    for position, row_index in enumerate(indices.detach().cpu().tolist()):
        row = rows[int(row_index)]
        by_unit.setdefault(row["unit"], []).append((float(row["endpoint"]), position))
    for values in by_unit.values():
        values.sort()
        for (_, earlier), (_, later) in zip(values, values[1:]):
            penalties.append(torch.relu(predictions[later] - predictions[earlier]))
    if not penalties:
        return predictions.new_zeros(())
    return torch.stack(penalties).mean()


def apply_validation_calibration(validation: pd.DataFrame,
                                 test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Fit bounded affine maps on validation predictions only."""
    from sklearn.linear_model import Ridge

    validation = validation.copy(); test = test.copy(); records = []
    calibrated = validation.model.eq("d2pred_expert_age_calibrated_v1")
    keys = validation.loc[calibrated, ["outer_fold", "seed", "group"]].drop_duplicates()
    for key in keys.itertuples(index=False):
        vm = calibrated & validation.outer_fold.eq(key.outer_fold) & validation.seed.eq(key.seed) & validation.group.eq(key.group)
        tm = test.model.eq("d2pred_expert_age_calibrated_v1") & test.outer_fold.eq(key.outer_fold) & test.seed.eq(key.seed) & test.group.eq(key.group)
        x = validation.loc[vm, "predicted_rul_cycles"].to_numpy().reshape(-1, 1)
        y = validation.loc[vm, "true_rul_cycles"].to_numpy()
        fit = Ridge(alpha=100.0).fit(x, y)
        slope = float(np.clip(fit.coef_[0], 0.75, 1.25))
        scale = float(validation.loc[vm, "target_scale_cycles"].iloc[0])
        intercept = float(np.clip(fit.intercept_, -0.20 * scale, 0.20 * scale))
        for frame, mask in ((validation, vm), (test, tm)):
            predicted = np.maximum(0.0, slope * frame.loc[mask, "predicted_rul_cycles"].to_numpy() + intercept)
            frame.loc[mask, "predicted_rul_cycles"] = predicted
            frame.loc[mask, "error_cycles"] = predicted - frame.loc[mask, "true_rul_cycles"].to_numpy()
            frame.loc[mask, "absolute_error_cycles"] = np.abs(frame.loc[mask, "error_cycles"])
        records.append({"outer_fold": key.outer_fold, "seed": key.seed, "group": key.group,
                        "slope": slope, "intercept_cycles": intercept,
                        "fit_split": "inner_validation_only", "ridge_alpha": 100.0})
    return validation, test, records


def predict(
    model: nn.Module, name: str, loader: DataLoader, rows: list[dict],
    target_scale: float, device: torch.device, fold: int, seed: int, role: str,
) -> list[dict]:
    model.eval()
    records = []
    with torch.inference_mode():
        for inputs, _, indices in loader:
            inputs = {key: value.to(device) for key, value in inputs.items()}
            values = forward_model(model, name, inputs).cpu().numpy() * target_scale
            for value, index in zip(values, indices.numpy()):
                row = rows[int(index)]
                records.append({
                    "outer_fold": fold, "seed": seed, "model": name, "role": role,
                    "unit": row["unit"], "group": row["group"],
                    "endpoint": row["endpoint"], "prefix_length": row["prefix_length"],
                    "true_rul_cycles": row["true_rul_cycles"],
                    "predicted_rul_cycles": float(value),
                    "error_cycles": float(value - row["true_rul_cycles"]),
                    "absolute_error_cycles": abs(float(value - row["true_rul_cycles"])),
                    "target_scale_cycles": target_scale,
                })
    return records


def train_one(
    rows: list[dict], name: str, fold: int, seed: int, out: Path,
    device: torch.device, epochs: int, batch_size: int,
) -> tuple[list[dict], list[dict], dict]:
    seed_everything(seed)
    rows = variant_rows(rows, name)
    parts = {role: [row for row in rows if row["split"] == role]
             for role in ("train", "validation", "test")}
    train_units = {row["unit"] for row in parts["train"]}
    # RUL at the first recorded cycle is lifetime-1 in the supplied table.
    target_scale = max(row["true_rul_cycles"] for row in parts["train"]) + 1.0
    normalization = fit_normalization(rows)
    loaders = {
        role: DataLoader(
            PrefixDataset(values, normalization, target_scale, name), batch_size=batch_size,
            shuffle=role == "train", num_workers=0,
        )
        for role, values in parts.items()
    }
    model = make_model(name).to(device)
    if (isinstance(model, ExpertExactD2Pred) and model.trend_fusion is not None
            and getattr(model, "trend_loss_active", True)):
        # The trend-fusion checkpoint is condition-specific: ACSim and ACCom
        # use their own independently fitted decompositions, while ACAll uses
        # its pooled ACAll decomposition.  Never force a group-specific run
        # to consume an ACAll checkpoint.
        groups = {str(row["group"]) for row in rows}
        fusion_group = "ACAll" if groups == {"ACAll"} else next(iter(groups))
        fusion_path = out / "fusion" / f"outer_fold_{fold}" / fusion_group / "fusion_checkpoint.pt"
        if not fusion_path.exists():
            raise RuntimeError(f"Missing condition-specific trend-fusion checkpoint: {fusion_path}")
        fusion_checkpoint = torch.load(fusion_path, map_location=device, weights_only=False)
        model.trend_fusion.load_state_dict(fusion_checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best = (float("inf"), None, 0)
    patience = 0
    log = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if epoch == 26:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = 5e-4
        losses = {}
        for role in ("train", "validation"):
            model.train(role == "train")
            values = []
            context = torch.enable_grad() if role == "train" else torch.no_grad()
            with context:
                for inputs, labels, indices in loaders[role]:
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                    labels = labels.to(device)
                    if role == "train":
                        optimizer.zero_grad(set_to_none=True)
                    if is_expert(name):
                        details = model(inputs, return_details=True)
                        loss = criterion(details["prediction"], labels)
                        if getattr(model, "gradual_loss_active", True):
                            loss = loss + 0.1 * criterion(details["gradual_prediction"], labels)
                        if getattr(model, "event_loss_active", True):
                            loss = loss + 0.1 * criterion(details["event_prediction"], labels)
                        if details["trend_fusion_details"] is None and getattr(model, "trend_loss_active", True):
                            raise RuntimeError("Strict expert model did not expose trainable trend fusion")
                        if getattr(model, "trend_loss_active", True):
                            fusion_loss, _ = stage_aware_fusion_loss(
                                details["fused_dwc"], inputs["di"], inputs["trend_candidates"],
                                inputs["sequence_mask"], details["trend_fusion_details"],
                                model.trend_fusion.config,
                            )
                            loss = loss + 0.1 * fusion_loss
                        if uses_late_loss(name):
                            late_weight = 1.0 + 0.75 * (1.0 - labels.detach()).clamp(0.0, 1.0)
                            loss = loss + 0.25 * (late_weight * (details["prediction"] - labels) ** 2).mean()
                        if uses_mono_loss(name):
                            loss = loss + 0.05 * monotonic_pair_loss(
                                details["prediction"], indices, parts[role]
                            )
                    else:
                        loss = criterion(forward_model(model, name, inputs), labels)
                    if role == "train":
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"Nonfinite training objective: {name}, fold {fold}, seed {seed}")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                    values.append(float(loss.item()))
            losses[role] = float(np.mean(values))
        log.append({"epoch": epoch, "train_mse": losses["train"],
                    "validation_mse": losses["validation"],
                    "learning_rate": optimizer.param_groups[0]["lr"]})
        if losses["validation"] < best[0] - 1e-8:
            best = (losses["validation"], deepcopy(model.state_dict()), epoch)
            patience = 0
        elif epoch > 25:
            patience += 1
        else:
            patience = 0
        if epoch > 25 and patience >= 8:
            break
    model.load_state_dict(best[1])
    elapsed = time.perf_counter() - started
    run_dir = out / "runs" / f"outer_fold_{fold}" / f"seed_{seed}" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(run_dir / "training_log.csv", index=False)
    architecture = (
        model.architecture_metadata() if callable(getattr(model, "architecture_metadata", None))
        else {"implementation": type(model).__name__, "source": "models.benchmark"}
    )
    metadata = {
        "outer_fold": fold, "seed": seed, "model": name,
        "target_scale_cycles": target_scale,
        "target_scale_rule": "max initial RUL plus one over inner-training devices only",
        "target_scale_fit_units": sorted(train_units),
        "normalization": normalization,
        "best_epoch": best[2], "best_validation_mse": best[0],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": elapsed, "architecture": architecture,
        "test_used_for_selection": False,
    }
    torch.save({"model_state_dict": best[1], **metadata}, run_dir / "best_checkpoint.pt")
    write_json(run_dir / "metadata.json", metadata)
    validation = predict(model, name, loaders["validation"], parts["validation"],
                         target_scale, device, fold, seed, "validation")
    test = predict(model, name, loaders["test"], parts["test"],
                   target_scale, device, fold, seed, "test")
    return validation, test, metadata


def unit_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    keys = ["outer_fold", "seed", "model", "group", "unit"]
    for values, part in predictions.groupby(keys, sort=True):
        error = part.error_cycles.to_numpy(float)
        absolute = np.abs(error)
        records.append(dict(zip(keys, values)) | {
            "num_predictions": len(part),
            "RMSE_cycles": float(np.sqrt(np.mean(error ** 2))),
            "MAE_cycles": float(np.mean(absolute)),
            "P90AE_cycles": float(np.percentile(absolute, 90)),
            "bias_cycles": float(np.mean(error)),
        })
    return pd.DataFrame(records)


def bootstrap_ci(values: np.ndarray, seed: int = 20260915) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    draws = np.asarray([
        generator.choice(values, size=len(values), replace=True).mean() for _ in range(2000)
    ])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def summarize(unit: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (model, group), part in unit.groupby(["model", "group"], sort=True):
        # First average seeds for each physical device, then summarize devices.
        device = part.groupby("unit", as_index=False)[
            ["RMSE_cycles", "MAE_cycles", "P90AE_cycles", "bias_cycles"]
        ].mean()
        low, high = bootstrap_ci(device.RMSE_cycles.to_numpy(float))
        records.append({
            "model": model, "group": group, "num_test_devices": len(device),
            "device_mean_RMSE_cycles": float(device.RMSE_cycles.mean()),
            "RMSE_95CI_low": low, "RMSE_95CI_high": high,
            "device_mean_MAE_cycles": float(device.MAE_cycles.mean()),
            "device_mean_P90AE_cycles": float(device.P90AE_cycles.mean()),
            "device_mean_bias_cycles": float(device.bias_cycles.mean()),
        })
    return pd.DataFrame(records).sort_values(["group", "device_mean_RMSE_cycles"])


def paired_tests(unit: pd.DataFrame) -> pd.DataFrame:
    reference = "d2pred_expert_exact_v1"
    metric_map = {"RMSE": "RMSE_cycles", "MAE": "MAE_cycles",
                  "P90AE": "P90AE_cycles"}
    records = []
    averaged = unit.groupby(["model", "group", "unit"], as_index=False)[
        list(metric_map.values())].mean()
    for group in sorted(averaged.group.unique()):
        group_values = averaged[averaged.group.eq(group)]
        for metric, column in metric_map.items():
            pivot = group_values.pivot(index="unit", columns="model", values=column)
            if reference not in pivot:
                continue
            comparisons = [m for m in pivot.columns if m != reference]
            family = []
            for model in comparisons:
                pair = pivot[[reference, model]].dropna()
                difference = pair[model] - pair[reference]
                try:
                    statistic, p_value = wilcoxon(difference, alternative="two-sided")
                except ValueError:
                    statistic, p_value = 0.0, 1.0
                family.append({
                    "group": group, "metric": metric, "reference": reference,
                    "comparison": model, "num_paired_devices": len(pair),
                    "mean_difference_comparison_minus_reference": float(difference.mean()),
                    "wilcoxon_statistic": float(statistic),
                    "p_value_two_sided": float(p_value),
                })
            order = sorted(range(len(family)), key=lambda i: family[i]["p_value_two_sided"])
            m = len(family)
            adjusted = [1.0] * m
            running = 0.0
            for rank, idx in enumerate(order):
                value = min(1.0, (m - rank) * family[idx]["p_value_two_sided"])
                running = max(running, value)
                adjusted[idx] = running
            for item, holm in zip(family, adjusted):
                item["holm_p_value"] = float(holm)
                item["holm_significant_0.05"] = bool(holm < 0.05)
                records.append(item)
    return pd.DataFrame(records)


def reliability(validation: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["outer_fold", "model", "unit", "group", "endpoint"]
    validation_ensemble = validation.groupby(keys, as_index=False).agg(
        true_rul_cycles=("true_rul_cycles", "first"),
        ensemble_prediction=("predicted_rul_cycles", "mean"),
    )
    validation_ensemble["absolute_error"] = abs(
        validation_ensemble.ensemble_prediction - validation_ensemble.true_rul_cycles
    )
    radii = validation_ensemble.groupby(["outer_fold", "model"]).absolute_error.quantile(.95)
    test_ensemble = test.groupby(keys, as_index=False).agg(
        true_rul_cycles=("true_rul_cycles", "first"),
        ensemble_prediction=("predicted_rul_cycles", "mean"),
        seed_std=("predicted_rul_cycles", "std"),
    )
    test_ensemble["calibration_radius"] = [
        radii.loc[(row.outer_fold, row.model)] for row in test_ensemble.itertuples()
    ]
    test_ensemble["interval_low"] = (
        test_ensemble.ensemble_prediction - test_ensemble.calibration_radius
    ).clip(lower=0)
    test_ensemble["interval_high"] = (
        test_ensemble.ensemble_prediction + test_ensemble.calibration_radius
    )
    test_ensemble["covered"] = (
        (test_ensemble.true_rul_cycles >= test_ensemble.interval_low)
        & (test_ensemble.true_rul_cycles <= test_ensemble.interval_high)
    )
    summary = test_ensemble.groupby(["model", "group"], as_index=False).agg(
        empirical_coverage=("covered", "mean"),
        mean_interval_width=("calibration_radius", lambda x: 2 * x.mean()),
        mean_seed_std=("seed_std", "mean"), num_predictions=("covered", "size"),
    )
    return test_ensemble, summary


def checksum_manifest(out: Path) -> None:
    records = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "checksum_manifest.csv":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            records.append({"path": str(path.relative_to(out)), "sha256": digest,
                            "bytes": path.stat().st_size})
    pd.DataFrame(records).to_csv(out / "checksum_manifest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--profile", choices=("core", "full"), default="core")
    parser.add_argument(
        "--models",
        default="",
        help="Optional comma-separated model subset; names must belong to the selected profile.",
    )
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--fusion-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--drop-units", default="", help="Comma-separated AC units to remove after the standard exclusions.")
    parser.add_argument("--merge-units-to-accom", default="", help="Comma-separated AC units whose group is reassigned to ACCom.")
    parser.add_argument("--mix-groups", action="store_true", help="Treat all retained AC devices as one ACAll group.")
    parser.add_argument("--only-group", choices=("ACSim", "ACCom"), default=None,
                        help="Restrict the run to one registered AC group while retaining its predefined device folds.")
    parser.add_argument("--group-manifest", type=Path, default=None,
                        help="Alternative unit/group manifest; defaults to the historical behavior manifest.")
    parser.add_argument("--folds-file", type=Path, default=None,
                        help="Reviewer-approved predefined nested device folds; validated before fitting.")
    parser.add_argument("--keep-all-units", action="store_true",
                        help="Do not apply the historical exclusion list (needed for the fixed-rule 38-unit run).")
    parser.add_argument("--aep-sensitivity", type=float, default=2.5)
    parser.add_argument("--aep-min-size", type=int, default=8)
    parser.add_argument("--aep-max-points", type=int, default=15)
    parser.add_argument("--aep-max-tokens", type=int, default=16)
    parser.add_argument("--prefix-start", type=int, default=30)
    parser.add_argument("--prefix-stride", type=int, default=20)
    parser.add_argument("--max-prefixes", type=int, default=20)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    seeds = tuple(int(value) for value in args.seeds.split(","))
    profile_models = tuple(dict.fromkeys(CORE_MODELS if args.profile == "core" else FULL_MODELS))
    if args.models:
        models = tuple(dict.fromkeys(value.strip() for value in args.models.split(",") if value.strip()))
        unknown = sorted(set(models) - set(profile_models))
        if unknown:
            parser.error(f"Models are not available in profile {args.profile}: {unknown}")
    else:
        models = profile_models
    if any(name.startswith(("ablation_", "controlled_ablation_")) for name in models):
        parser.error("Ablation aliases are disabled pending historical implementation review.")
    drop_units = {int(v) for v in args.drop_units.split(",") if v.strip()}
    merge_units = {int(v) for v in args.merge_units_to_accom.split(",") if v.strip()}
    group_manifest = args.group_manifest
    if group_manifest is not None and not group_manifest.is_absolute():
        group_manifest = ROOT / group_manifest
    frame = load_ac(drop_units=drop_units, merge_units_to_accom=merge_units,
                    group_manifest=group_manifest,
                    keep_standard_exclusions=not args.keep_all_units)
    if args.mix_groups:
        frame["group"] = "ACAll"
    folds_file = args.folds_file
    if folds_file is not None and not folds_file.is_absolute():
        folds_file = ROOT / folds_file
    if folds_file is not None:
        folds = json.loads(folds_file.read_text(encoding="utf-8"))
        if args.only_group:
            selected = set(frame.loc[frame.group.eq(args.only_group), "unit"].astype(int))
            frame = frame[frame.unit.isin(selected)].copy()
            filtered = []
            for fold in folds:
                item = {"outer_fold": fold["outer_fold"]}
                for role in ("train", "validation", "test"):
                    item[f"{role}_units"] = sorted(set(fold[f"{role}_units"]) & selected)
                filtered.append(item)
            folds = validate_predefined_folds(frame, filtered)
        else:
            folds = validate_predefined_folds(frame, folds)
        if len(folds) != args.outer_folds:
            parser.error("--outer-folds must match the number of predefined folds")
    else:
        folds = make_folds(frame, args.outer_folds)
    protocol = {
        "version": "ac_nested_gru_v2", "created": now(),
        "historical_reference": {"ACSim_device_mean_RMSE_percent": 5.750009369687675,
                                 "ACCom_device_mean_RMSE_percent": 5.1296693438438385,
                                 "final_result": False},
        "retained_units": sorted(frame.unit.unique().tolist()),
        "scenario": {"drop_units": sorted(drop_units), "merge_units_to_accom": sorted(merge_units),
                     "group_manifest": str(group_manifest) if group_manifest else None,
                     "predefined_folds": str(folds_file) if folds_file else None,
                     "historical_standard_exclusions_applied": not args.keep_all_units},
        "excluded_units": sorted(drop_units if args.keep_all_units else EXCLUDED_UNITS | drop_units),
        "groups": frame.groupby("group").unit.nunique().to_dict(),
        "outer_folds": args.outer_folds, "inner_validation": "device-wise stratified 22% of outer development devices",
        "seeds": seeds, "models": models,
        "target": "RUL cycles divided by train-only maximum initial RUL plus one",
        "test_device_lifetime_used_for_target_or_inverse_transform": False,
        "decomposition": "prefix-causal fold-fitted stage-aware Wiener/STL/AE fusion",
        "shared_group_blind_decomposition": False,
        "decomposition_protocol": "ACSim and ACCom use independent DWC/EDC fits; ACAll uses its own pooled ACAll DWC/EDC fit. No cross-condition decomposition is shared.",
        "aep": {"sensitivity": args.aep_sensitivity,
                "minimum_segment_length": args.aep_min_size,
                "maximum_change_points": args.aep_max_points,
                "maximum_tokens": args.aep_max_tokens,
                "descriptor_dimension": 12},
        "prefix_generation": {"start": args.prefix_start,
                              "stride": args.prefix_stride,
                              "maximum_prefixes": args.max_prefixes},
        "training": {"optimizer": "Adam", "learning_rate": 1e-3,
                     "batch_size": args.batch_size, "maximum_epochs": args.epochs,
                     "early_stopping_patience": 8, "fusion_epochs": args.fusion_epochs},
        "primary_metric": "device_mean_RMSE_cycles after averaging seeds per device",
    }
    write_json(out / "protocol.json", protocol)
    write_json(out / "folds.json", folds)
    write_json(out / "run_status.json", {"status": "running", "started": now()})
    all_validation, all_test, runtimes = [], [], []
    for fold in folds:
        print(f"[fold {fold['outer_fold']}] fitting causal decomposition", flush=True)
        rows = build_fold_rows(frame, fold, out, device, args.fusion_epochs,
                               args.mix_groups,
                               {"sensitivity": args.aep_sensitivity,
                                "min_size": args.aep_min_size,
                                "max_points": args.aep_max_points,
                                "max_tokens": args.aep_max_tokens},
                               {"start": args.prefix_start,
                                "stride": args.prefix_stride,
                                "maximum": args.max_prefixes})
        if max(row["edc_identity_error"] for row in rows) > 1e-6:
            raise AssertionError("EDC != DI-DWC")
        for seed in seeds:
            for name in models:
                print(f"[fold {fold['outer_fold']}][seed {seed}][{name}]", flush=True)
                validation, test, metadata = train_one(
                    rows, name, fold["outer_fold"], seed, out, device,
                    args.epochs, args.batch_size,
                )
                all_validation.extend(validation)
                all_test.extend(test)
                runtimes.append({key: metadata[key] for key in (
                    "outer_fold", "seed", "model", "best_epoch", "best_validation_mse",
                    "parameter_count", "training_seconds", "target_scale_cycles",
                )})
    validation = pd.DataFrame(all_validation)
    test = pd.DataFrame(all_test)
    validation, test, calibration = apply_validation_calibration(validation, test)
    pd.DataFrame(calibration).to_csv(out / "validation_calibration.csv", index=False)
    validation.to_csv(out / "validation_predictions.csv", index=False)
    test.to_csv(out / "test_predictions.csv", index=False)
    unit = unit_metrics(test)
    unit.to_csv(out / "device_level_metrics.csv", index=False)
    summary = summarize(unit)
    summary.to_csv(out / "summary_metrics.csv", index=False)
    paired_tests(unit).to_csv(out / "paired_statistical_tests.csv", index=False)
    ensemble, reliability_summary = reliability(validation, test)
    ensemble.to_csv(out / "seed_ensemble_predictions_with_intervals.csv", index=False)
    reliability_summary.to_csv(out / "reliability_summary.csv", index=False)
    pd.DataFrame(runtimes).to_csv(out / "computational_cost.csv", index=False)
    write_json(out / "environment.json", {
        "timestamp": now(), "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "command": " ".join(os.sys.argv),
    })
    write_json(out / "run_status.json", {"status": "verified_complete", "completed": now(),
                                          "num_test_predictions": len(test)})
    checksum_manifest(out)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
