#!/usr/bin/env python3
"""Leakage-safe nested device-wise AC rerun for the frozen GRU D2Pred model.

The historical 5.75/5.13 results are retained only as a reference.  This
runner removes device-specific lifetime normalization: every predictor target
scale is fitted from the inner-training devices of its outer fold.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
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
    DECOMPOSITION, STABLE, endpoints, train_fusion,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results/ac_nested_gru_final"
EXCLUDED_UNITS = {1, 3, 11, 35, 38}
SEEDS = (17, 42, 73)
SENSITIVITY = {"gamma": 2.5, "min_size": 8, "max_points": 15, "max_tokens": 16,
               "prefix_start": 30, "prefix_stride": 20, "max_prefixes": 20}
HEALTHY_OVERRIDE = {}
HEALTHY_METHOD = "current"
HEALTHY_PARAMS = {}
WIENER_MODE = "standard"
REGIME_MODE = "fixed"
REGIME_ALPHA = 0.5
CORE_MODELS = (
    "gru_d2pred_v2", "d2pred_expert_exact_v1", "d2pred_expert_age_v1",
    "d2pred_expert_cycle_v1", "d2pred_expert_prefix_v1",
    "d2pred_expert_age_calibrated_v1", "d2pred_expert_age_gate_v1",
    "d2pred_expert_age_gate_late_v1", "d2pred_expert_age_gate_late_mono_v1",
    "CNN", "Transformer",
    "TCN_BiGRU", "KGHM", "ablation_no_dwc", "ablation_no_edc",
    "ablation_no_tokens", "ablation_no_fusion", "ablation_no_decomposition",
    "ablation_no_trend_fusion", "ablation_no_gradual_encoder",
    "ablation_no_token_projection", "ablation_no_event_encoder",
    "ablation_no_guided_fusion", "ablation_no_joint_loss",
    "ablation_no_wiener", "ablation_no_stl", "ablation_no_autoencoder_fusion",
    "ablation_preprocess_no_dwc", "ablation_preprocess_no_edc",
    "ablation_preprocess_no_decomposition", "ablation_no_aep",
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


def event_tokens(edc: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    points = causal_change_points(edc, sensitivity=SENSITIVITY["gamma"],
                                  min_size=SENSITIVITY["min_size"], max_points=SENSITIVITY["max_points"])
    token_values = segment_tokens(edc, points, mode="stats")
    tokens, mask = pad_tokens(token_values, SENSITIVITY["max_tokens"])
    return tokens.astype(np.float32), mask.astype(bool), points


def build_fold_rows(
    frame: pd.DataFrame,
    fold: dict,
    out: Path,
    device: torch.device,
    fusion_epochs: int,
    mix_groups: bool = False,
    shared_decomposition: bool = False,
) -> list[dict]:
    cache = out / "cache" / f"outer_fold_{fold['outer_fold']}_rows.pt"
    if cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        expected_cache_version = 10 if REGIME_MODE != "fixed" else (3 if shared_decomposition else 2)
        if payload.get("cache_version") == expected_cache_version:
            return payload["rows"]
    assigned = split_frame(frame, fold)
    models = {}
    if REGIME_MODE in {"fixed_grouped", "prefix_gated", "shared_specific"}:
        group_names = ("ACAll", "ACCom", "ACSim")
    else:
        group_names = ("ACAll",) if (mix_groups or shared_decomposition) else ("ACCom", "ACSim")
    present_groups = set(assigned.group.astype(str).unique())
    group_names = tuple(group for group in group_names if group in present_groups or group == "ACAll")
    stable_config = {"ACAll": STABLE["ACCom"], **STABLE}
    if HEALTHY_OVERRIDE:
        stable_config = {
            key: replace(value, **HEALTHY_OVERRIDE) for key, value in stable_config.items()
        }
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
    # Prefix-gating threshold is estimated from training devices only.
    gate_threshold = 0.0
    if REGIME_MODE == "prefix_gated":
        feats = []
        for unit, uf in assigned[assigned.split.eq("train")].groupby("unit"):
            values = uf.sort_values("cycle").DI.to_numpy(np.float32)
            n = min(len(values), 20)
            feats.append((float(np.polyfit(np.arange(n), values[:n], 1)[0]) if n >= 2 else 0.0,
                          str(uf.group.iloc[0])))
        by_group = {g: [v for v, label in feats if label == g] for g in ("ACSim", "ACCom")}
        if by_group["ACSim"] and by_group["ACCom"]:
            gate_threshold = float((np.median(by_group["ACSim"]) + np.median(by_group["ACCom"])) / 2.0)
        elif feats:
            gate_threshold = float(np.median([v for v, _ in feats]))
    rows = []
    for unit, unit_frame in assigned.groupby("unit", sort=True):
        unit_frame = unit_frame.sort_values("cycle")
        di_full = unit_frame.DI.to_numpy(np.float32)
        group = str(unit_frame.group.iloc[0])
        role = str(unit_frame.split.iloc[0])
        length = len(unit_frame)
        values = list(range(min(SENSITIVITY["prefix_start"], length), length + 1, SENSITIVITY["prefix_stride"]))
        if not values or values[-1] != length:
            values.append(length)
        if len(values) > SENSITIVITY["max_prefixes"]:
            indices = np.linspace(0, len(values) - 1, SENSITIVITY["max_prefixes"]).round().astype(int)
            values = [values[index] for index in indices]
        for end in values:
            observed = unit_frame.iloc[:end]
            di = di_full[:end]
            if REGIME_MODE == "prefix_gated":
                n_gate = min(len(di), 20)
                slope = float(np.polyfit(np.arange(n_gate), di[:n_gate], 1)[0]) if n_gate >= 2 else 0.0
                decomposition_group = "ACCom" if slope >= gate_threshold else "ACSim"
            else:
                decomposition_group = "ACAll" if shared_decomposition else group
            dwc, edc, metadata = stage_aware_causal_decompose(
                di, models[decomposition_group], stable_config[decomposition_group],
                DECOMPOSITION, device, return_candidates=True,
                healthy_method=HEALTHY_METHOD, healthy_params=HEALTHY_PARAMS,
                wiener_mode=WIENER_MODE,
            )
            if REGIME_MODE == "shared_specific":
                shared_dwc, _, shared_meta = stage_aware_causal_decompose(
                    di, models["ACAll"], stable_config["ACAll"], DECOMPOSITION, device,
                    return_candidates=True, healthy_method=HEALTHY_METHOD, healthy_params=HEALTHY_PARAMS,
                )
                dwc = (shared_dwc + REGIME_ALPHA * (dwc - shared_dwc)).astype(np.float32)
                edc = (di - dwc).astype(np.float32)
                metadata = {**metadata, "shared_specific_alpha": REGIME_ALPHA,
                            "shared_wiener": shared_meta.get("wiener", {})}
            tokens, mask, points = event_tokens(edc)
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
                "healthy_detector": str(metadata.get("wiener", {}).get("detector", HEALTHY_METHOD)),
                "healthy_stage_end": int(metadata.get("wiener", {}).get("stable_end", 0)),
                "wiener_fit_len": int(metadata.get("wiener", {}).get("fit_len", 0)),
                "wiener_beta_0": float(metadata.get("wiener", {}).get("beta_0", np.nan)),
                "wiener_mu": float(metadata.get("wiener", {}).get("mu", np.nan)),
                "wiener_sigma": float(metadata.get("wiener", {}).get("sigma", np.nan)),
                "wiener_fit_rmse": float(metadata.get("wiener", {}).get("fit_rmse", np.nan)),
                "tokens": tokens, "token_mask": mask,
                "change_points": points,
                "edc_identity_error": float(metadata["edc_identity_max_abs_error"]),
            })
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cache_version": 10 if REGIME_MODE != "fixed" else (3 if shared_decomposition else 2),
                "fold": fold, "rows": rows}, cache)
    diagnostic_rows = []
    for row in rows:
        diagnostic_rows.append({key: row.get(key) for key in (
            "unit", "group", "split", "prefix_length", "healthy_detector",
            "healthy_stage_end", "wiener_fit_len", "wiener_beta_0", "wiener_mu",
            "wiener_sigma", "wiener_fit_rmse", "stable_fraction", "wiener_reliability",
            "change_points")})
    diagnostic_dir = out / "diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnostic_rows).to_csv(
        diagnostic_dir / f"outer_fold_{fold['outer_fold']}_healthy_stage.csv", index=False
    )
    return rows


def variant_rows(rows: list[dict], model_name: str) -> list[dict]:
    result = []
    for source in rows:
        row = {**source}
        if model_name in {"ablation_no_wiener", "ablation_no_stl", "ablation_no_autoencoder_fusion"}:
            candidates = source["trend_candidates"]
            if model_name == "ablation_no_wiener":
                row["dwc"] = candidates[1].copy()
            elif model_name == "ablation_no_stl":
                row["dwc"] = candidates[0].copy()
            else:
                row["dwc"] = candidates.mean(axis=0).astype(np.float32)
            row["edc"] = (source["di"] - row["dwc"]).astype(np.float32)
            row["tokens"], row["token_mask"], row["change_points"] = event_tokens(row["edc"])
        elif model_name == "ablation_preprocess_no_dwc":
            row["dwc"] = np.zeros_like(source["dwc"])
            row["trend_candidates"] = np.zeros_like(source["trend_candidates"])
        elif model_name == "ablation_preprocess_no_edc":
            row["edc"] = np.zeros_like(source["edc"])
            row["tokens"] = np.zeros_like(source["tokens"])
            row["token_mask"] = np.zeros_like(source["token_mask"])
            row["token_mask"][0] = True
            row["change_points"] = []
        elif model_name in {"ablation_no_decomposition", "ablation_preprocess_no_decomposition"}:
            row["dwc"] = source["di"].copy()
            row["edc"] = np.zeros_like(source["edc"])
            row["tokens"] = np.zeros_like(source["tokens"])
            row["token_mask"] = np.zeros_like(source["token_mask"])
            row["token_mask"][0] = True
            row["trend_candidates"] = np.stack((source["di"], source["di"])).astype(np.float32)
            row["change_points"] = []
        elif model_name == "ablation_no_aep":
            length = len(source["edc"])
            fixed_points = sorted({max(1, min(length - 1, int(round(length * q)))) for q in (.25, .5, .75)})
            token_values = segment_tokens(source["edc"], fixed_points, mode="stats")
            row["tokens"], row["token_mask"] = pad_tokens(token_values, 16)
            row["token_mask"] = row["token_mask"].astype(bool)
            row["change_points"] = fixed_points
        result.append(row)
    return result


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
    result["token_mean"] = valid.mean(axis=0).tolist()
    std = valid.std(axis=0)
    std[std < 1e-6] = 1.0
    result["token_std"] = std.tolist()
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


def make_model(name: str) -> nn.Module:
    if name == "gru_d2pred_v2":
        return FrozenGRUD2Pred()
    if name == "d2pred_expert_exact_v1":
        return ExpertExactD2Pred(maximum_tokens=SENSITIVITY["max_tokens"])
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
    strict = {
        "ablation_no_trend_fusion": {"joint_trend_fusion": False},
        "ablation_no_gradual_encoder": {"use_gradual_encoder": False},
        "ablation_no_token_projection": {"use_token_projection": False},
        "ablation_no_event_encoder": {"use_event_encoder": False},
        "ablation_no_guided_fusion": {"use_guided_fusion": False},
    }
    if name in strict:
        return ExpertExactD2Pred(maximum_tokens=SENSITIVITY["max_tokens"], **strict[name])
    if name == "ablation_no_joint_loss":
        return ExpertExactD2Pred(maximum_tokens=SENSITIVITY["max_tokens"])
    if name in {
        "ablation_no_wiener", "ablation_no_stl", "ablation_no_autoencoder_fusion",
        "ablation_preprocess_no_dwc", "ablation_preprocess_no_decomposition",
    }:
        return ExpertExactD2Pred(maximum_tokens=SENSITIVITY["max_tokens"], joint_trend_fusion=False)
    if name in {"ablation_preprocess_no_edc", "ablation_no_aep"}:
        return ExpertExactD2Pred(maximum_tokens=SENSITIVITY["max_tokens"])
    # Legacy names are retained for provenance but are not part of the strict run.
    if name == "ablation_no_dwc":
        return FrozenGRUD2Pred(use_dwc=False)
    if name == "ablation_no_edc":
        return FrozenGRUD2Pred(use_edc=False, use_tokens=False)
    if name == "ablation_no_tokens":
        return FrozenGRUD2Pred(use_tokens=False)
    if name == "ablation_no_fusion":
        return FrozenGRUD2Pred(fusion="average_prediction")
    if name == "ablation_no_decomposition":
        return FrozenGRUD2Pred()
    return build_baseline(name)


def forward_model(model: nn.Module, name: str, inputs: dict[str, torch.Tensor]):
    if name.startswith("gru_") or name.startswith("d2pred_") or name.startswith("ablation_"):
        return model(inputs)
    if name == "KGHM":
        return model(torch.stack((inputs["dwc"], inputs["edc"]), dim=-1))
    return model(inputs["di"])


def is_expert(name: str) -> bool:
    return name.startswith("d2pred_expert_") or name in {
        "ablation_no_trend_fusion", "ablation_no_gradual_encoder",
        "ablation_no_token_projection", "ablation_no_event_encoder",
        "ablation_no_guided_fusion", "ablation_no_joint_loss",
        "ablation_no_wiener", "ablation_no_stl", "ablation_no_autoencoder_fusion",
        "ablation_preprocess_no_dwc", "ablation_preprocess_no_edc",
        "ablation_preprocess_no_decomposition", "ablation_no_aep",
    }


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
    fusion_checkpoint_path: Path | None = None,
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
    if isinstance(model, ExpertExactD2Pred) and model.trend_fusion is not None:
        fusion_path = fusion_checkpoint_path or (
            out / "fusion" / f"outer_fold_{fold}" / "ACAll" / "fusion_checkpoint.pt"
        )
        if not fusion_path.exists():
            raise RuntimeError("Strict expert joint fusion requires the shared ACAll fusion checkpoint")
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
                        if name != "ablation_no_joint_loss":
                            loss = loss + 0.1 * criterion(details["gradual_prediction"], labels)
                            loss = loss + 0.1 * criterion(details["event_prediction"], labels)
                            if details["trend_fusion_details"] is not None:
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
    averaged = unit.groupby(["model", "group", "unit"], as_index=False).RMSE_cycles.mean()
    records = []
    for group, group_values in averaged.groupby("group"):
        pivot = group_values.pivot(index="unit", columns="model", values="RMSE_cycles")
        if "gru_d2pred_v2" not in pivot:
            continue
        for model in pivot.columns:
            if model == "gru_d2pred_v2":
                continue
            pair = pivot[["gru_d2pred_v2", model]].dropna()
            difference = pair[model] - pair["gru_d2pred_v2"]
            try:
                statistic, p_value = wilcoxon(difference)
            except ValueError:
                statistic, p_value = 0.0, 1.0
            records.append({
                "group": group, "reference": "gru_d2pred_v2", "comparison": model,
                "num_paired_devices": len(pair),
                "mean_RMSE_difference_comparison_minus_main": float(difference.mean()),
                "wilcoxon_statistic": float(statistic), "p_value": float(p_value),
                "paired_cohen_dz": float(difference.mean() / difference.std(ddof=1))
                if len(difference) > 1 and difference.std(ddof=1) > 0 else 0.0,
            })
    result = pd.DataFrame(records)
    if len(result):
        result["holm_threshold_rank"] = result.p_value.rank(method="first").astype(int)
        result["holm_significant_0.05"] = (
            result.p_value <= 0.05 / (len(result) - result.holm_threshold_rank + 1)
        )
    return result


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
    global HEALTHY_METHOD, HEALTHY_PARAMS, REGIME_MODE, REGIME_ALPHA, WIENER_MODE
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
                        help="Run an independently trained single-group AC experiment.")
    parser.add_argument("--aep-gamma", type=float, default=2.5)
    parser.add_argument("--aep-min-size", type=int, default=8)
    parser.add_argument("--aep-max-points", type=int, default=15)
    parser.add_argument("--aep-max-tokens", type=int, default=16)
    parser.add_argument("--prefix-start-cycle", type=int, default=30)
    parser.add_argument("--prefix-stride", type=int, default=20)
    parser.add_argument("--maximum-prefixes", type=int, default=20)
    parser.add_argument("--healthy-k-min", type=int, default=None)
    parser.add_argument("--healthy-gamma-r", type=float, default=None)
    parser.add_argument("--healthy-consecutive-q", type=int, default=None)
    parser.add_argument("--healthy-method", choices=("current", "increment_rate"), default="current")
    parser.add_argument("--wiener-mode", choices=("standard", "reliability_weighted", "online_local_drift", "piecewise_linear"), default="standard")
    parser.add_argument("--wiener-local-window", type=int, default=12)
    parser.add_argument("--healthy-initial-window", type=int, default=20)
    parser.add_argument("--healthy-rate-mad-multiplier", type=float, default=3.0)
    parser.add_argument("--healthy-min-rate-sign-consistency", type=float, default=0.8)
    parser.add_argument("--regime-mode", choices=("fixed", "fixed_grouped", "prefix_gated", "shared_specific"), default="fixed",
                        help="Isolated A/B/C ACAll decomposition mode.")
    parser.add_argument("--regime-alpha", type=float, default=0.5,
                        help="Specific-branch weight for shared_specific mode.")
    parser.add_argument("--shared-decomposition", action="store_true",
                        help="Use one group-blind ACAll fusion while retaining ACCom/ACSim reporting labels.")
    parser.add_argument("--group-manifest", type=Path, default=None,
                        help="Alternative unit/group manifest; defaults to the historical behavior manifest.")
    parser.add_argument("--folds-file", type=Path, default=None,
                        help="Reviewer-approved predefined nested device folds; validated before fitting.")
    parser.add_argument("--keep-all-units", action="store_true",
                        help="Do not apply the historical exclusion list (needed for the fixed-rule 38-unit run).")
    args = parser.parse_args()
    SENSITIVITY.update({"gamma": args.aep_gamma, "min_size": args.aep_min_size,
                        "max_points": args.aep_max_points, "max_tokens": args.aep_max_tokens,
                        "prefix_start": args.prefix_start_cycle, "prefix_stride": args.prefix_stride,
                        "max_prefixes": args.maximum_prefixes})
    if args.healthy_k_min is not None:
        HEALTHY_OVERRIDE["k_min"] = args.healthy_k_min
    if args.healthy_gamma_r is not None:
        HEALTHY_OVERRIDE["gamma_r"] = args.healthy_gamma_r
    if args.healthy_consecutive_q is not None:
        HEALTHY_OVERRIDE["consecutive_q"] = args.healthy_consecutive_q
    HEALTHY_METHOD = args.healthy_method
    WIENER_MODE = args.wiener_mode
    if WIENER_MODE == "online_local_drift":
        HEALTHY_PARAMS["local_window"] = args.wiener_local_window
    REGIME_MODE = args.regime_mode
    REGIME_ALPHA = args.regime_alpha
    HEALTHY_PARAMS.update({"initial_window": args.healthy_initial_window,
                           "rate_mad_multiplier": args.healthy_rate_mad_multiplier,
                           "consecutive_violations": args.healthy_consecutive_q or 2,
                           "minimum_rate_sign_consistency": args.healthy_min_rate_sign_consistency})
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
    drop_units = {int(v) for v in args.drop_units.split(",") if v.strip()}
    merge_units = {int(v) for v in args.merge_units_to_accom.split(",") if v.strip()}
    group_manifest = args.group_manifest
    if group_manifest is not None and not group_manifest.is_absolute():
        group_manifest = ROOT / group_manifest
    frame = load_ac(drop_units=drop_units, merge_units_to_accom=merge_units,
                    group_manifest=group_manifest,
                    keep_standard_exclusions=not args.keep_all_units)
    if args.only_group is not None:
        frame = frame[frame.group.eq(args.only_group)].copy()
    if args.mix_groups:
        frame["group"] = "ACAll"
    folds_file = args.folds_file
    if folds_file is not None and not folds_file.is_absolute():
        folds_file = ROOT / folds_file
    if folds_file is not None:
        folds = validate_predefined_folds(frame, json.loads(folds_file.read_text(encoding="utf-8")))
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
                     "only_group": args.only_group,
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
        "shared_group_blind_decomposition": args.shared_decomposition,
        "aep": {"sensitivity": SENSITIVITY["gamma"], "minimum_segment_length": SENSITIVITY["min_size"],
                "maximum_change_points": SENSITIVITY["max_points"], "maximum_tokens": SENSITIVITY["max_tokens"],
                     "descriptor_dimension": 12, "sensitivity_override": dict(SENSITIVITY)},
        "healthy_stage_override": dict(HEALTHY_OVERRIDE),
        "healthy_stage_method": HEALTHY_METHOD,
        "wiener_mode": WIENER_MODE,
        "healthy_stage_method_params": dict(HEALTHY_PARAMS),
        "regime_mode": REGIME_MODE,
        "regime_alpha": REGIME_ALPHA,
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
                               args.mix_groups, args.shared_decomposition)
        if max(row["edc_identity_error"] for row in rows) > 1e-6:
            raise AssertionError("EDC != DI-DWC")
        for seed in seeds:
            for name in models:
                print(f"[fold {fold['outer_fold']}][seed {seed}][{name}]", flush=True)
                validation, test, metadata = train_one(
                    rows, name, fold["outer_fold"], seed, out, device,
                    args.epochs, args.batch_size,
                    fusion_checkpoint_path=(
                        out / "fusion" / f"outer_fold_{fold['outer_fold']}" /
                        (args.only_group if args.only_group is not None else "ACAll") /
                        "fusion_checkpoint.pt"
                    ),
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
