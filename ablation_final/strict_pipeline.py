"""Variant-isolated, fold-scoped preprocessing for strict AC ablations."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from analysis.causal_decomposition import freeze_fusion_model
from analysis.ac_cpm_core import pad_tokens, segment_tokens
from analysis.strict_ablation import fusion_config_for_variant, resolve_variant
from analysis.theory_aligned_decomposition import StageAwareTrendAutoencoder
from run_ac_nested_gru_rerun import endpoints, event_tokens, resize, split_frame
from run_stage_aware_aefusion_ac import DECOMPOSITION, STABLE, fusion_item, train_fusion


ROOT = Path(__file__).resolve().parents[1]
FULL_REFERENCE = ROOT / "results/frozen_protocol_final/comparison"
PREPROCESSING_VERSION = "strict_expert_exact_causal_preprocess_v4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def global_event_token(values: np.ndarray, maximum_tokens: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Build one whole-prefix statistics token without adaptive partitioning."""
    token = segment_tokens(values, change_points=(), mode="stats")
    tokens, mask = pad_tokens(token, maximum_tokens)
    return tokens.astype(np.float32), mask.astype(bool)


def _stable_for(dataset: str):
    return STABLE["ACCom"] if dataset == "ACAll" else STABLE[dataset]


def _fold_fusion_path(dataset: str, fold: int) -> Path:
    return FULL_REFERENCE / dataset / "fusion" / f"outer_fold_{fold}" / dataset / "fusion_checkpoint.pt"


def _load_fusion_checkpoint(path: Path, config, device: torch.device):
    model = StageAwareTrendAutoencoder(config=config).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return freeze_fusion_model(model)


def _fit_single_source_fusion(
    frame: pd.DataFrame,
    dataset: str,
    fold: dict,
    variant: str,
    output_root: Path,
    device: torch.device,
    epochs: int,
):
    config = fusion_config_for_variant(variant)
    if config is None or config.candidate_source not in {"wiener", "stl"}:
        raise ValueError(f"{variant} does not define a single-source fusion pretraining route")
    assigned = split_frame(frame, fold)
    selected = assigned.copy()
    if dataset == "ACAll":
        group_names = ("ACAll",)
        selected["group"] = "ACAll"
    else:
        group_names = (dataset,)
        selected = selected[selected.group.eq(dataset)].copy()
    selected = selected.rename(columns={"unit": "unit_id"})
    selected["split"] = selected.split.replace({"validation": "val"})
    model, logs, best_epoch = train_fusion(
        selected, _stable_for(dataset), device, epochs, fusion_config=config
    )
    checkpoint_path = (
        output_root / "pretraining" / dataset / variant
        / f"outer_fold_{fold['outer_fold']}" / dataset / "fusion_checkpoint.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": model.architecture_metadata(),
        "fusion_config": asdict(config),
        "fit_units": sorted(selected[selected.split.eq("train")].unit_id.unique().tolist()),
        "validation_units": sorted(selected[selected.split.eq("val")].unit_id.unique().tolist()),
        "test_units_used": False,
        "best_epoch": best_epoch,
    }, checkpoint_path)
    pd.DataFrame(logs).to_csv(checkpoint_path.parent / "training_log.csv", index=False)
    return model, checkpoint_path


def _prefix_components(
    di: np.ndarray,
    variant: str,
    fusion_model: StageAwareTrendAutoencoder | None,
    stable,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    spec = resolve_variant(variant)
    if not spec.decomposition:
        width = len(di)
        zeros = np.zeros(width, dtype=np.float32)
        candidates = np.zeros((2, width), dtype=np.float32)
        return di.copy(), di.copy(), zeros, candidates, 0.0, 1.0

    fusion_config = fusion_config_for_variant(variant)
    item = fusion_item(di, stable, fusion_config or fusion_config_for_variant("no_edc") or None)
    candidates = np.asarray(item["trends"], dtype=np.float32)
    stable_fraction = float(item["stable_fraction"])
    reliability = float(item["reliability"])

    if spec.trend_fusion_mode == "fixed_equal_average":
        dwc = candidates.mean(axis=0).astype(np.float32)
    else:
        if fusion_model is None:
            raise RuntimeError(f"Missing fold-pretrained trend model for {variant}")
        trend_tensor = torch.from_numpy(candidates)[None].to(device)
        mask = torch.ones((1, len(di)), dtype=torch.bool, device=device)
        with torch.inference_mode():
            output, _ = fusion_model.forward_components(
                trend_tensor, mask,
                torch.tensor([stable_fraction], dtype=torch.float32, device=device),
                torch.tensor([reliability], dtype=torch.float32, device=device),
            )
        dwc = output[0].cpu().numpy().astype(np.float32)
    edc = (di - dwc).astype(np.float32)
    return di.copy(), dwc, edc, candidates, stable_fraction, reliability


def build_variant_fold_rows(
    frame: pd.DataFrame,
    fold: dict,
    dataset: str,
    variant: str,
    output_root: Path,
    device: torch.device,
    *,
    fusion_epochs: int = 50,
    prefix_config: dict[str, int] | None = None,
    aep_config: dict[str, float | int] | None = None,
) -> tuple[list[dict], Path | None, Path]:
    """Build an isolated causal cache; no cache auto-discovery is performed."""
    spec = resolve_variant(variant)
    variant = spec.variant
    fold_id = int(fold["outer_fold"])
    source = ROOT / "data/industrial/DI_df.csv"
    folds_path = FULL_REFERENCE / dataset / "folds.json"
    preprocessing_sources = (
        Path(__file__).resolve(),
        ROOT / "analysis/strict_ablation.py",
        ROOT / "analysis/causal_decomposition.py",
        ROOT / "analysis/theory_aligned_decomposition.py",
        ROOT / "run_ac_nested_gru_rerun.py",
        ROOT / "run_stage_aware_aefusion_ac.py",
    )
    signature = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing_code_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in preprocessing_sources
        },
        "dataset": dataset,
        "variant": variant,
        "di_sha256": sha256(source),
        "fold_manifest_sha256": sha256(folds_path),
        "fold": fold_id,
        "prefix": prefix_config or {"start": 30, "stride": 20, "maximum": 20},
        "aep": aep_config or {"sensitivity": 2.5, "min_size": 8, "max_points": 15, "max_tokens": 16},
    }
    signature_key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache = (
        output_root / "cache" / dataset / variant
        / f"outer_fold_{fold_id}_{signature_key[:16]}.pt"
    )
    if cache.is_file():
        cached = torch.load(cache, map_location="cpu", weights_only=False)
        if cached.get("cache_version") == 3 and cached.get("cache_signature") == signature:
            fusion_path = _fold_fusion_path(dataset, fold_id)
            if spec.trend_fusion_mode in {"adaptive_stl_anchored_ae", "adaptive_wiener_anchored_ae"}:
                fusion_path = (
                    output_root / "pretraining" / dataset / variant
                    / f"outer_fold_{fold_id}" / dataset / "fusion_checkpoint.pt"
                )
            elif variant == "no_autoencoder_fusion" or variant == "no_decomposition":
                fusion_path = None
            return cached["rows"], fusion_path, cache
    fusion_model = None
    fusion_path = None
    if spec.trend_fusion_mode in {"adaptive_stl_anchored_ae", "adaptive_wiener_anchored_ae"}:
        fusion_model, fusion_path = _fit_single_source_fusion(
            frame, dataset, fold, variant, output_root, device, fusion_epochs
        )
    elif spec.joint_trend_fusion or variant == "no_dwc":
        fusion_path = _fold_fusion_path(dataset, fold_id)
        fusion_model = _load_fusion_checkpoint(
            fusion_path, fusion_config_for_variant(variant) or fusion_config_for_variant("no_edc"), device
        )
    elif spec.trend_fusion_mode == "fold_pretrained_adaptive_ae":
        fusion_path = _fold_fusion_path(dataset, fold_id)
        fusion_model = _load_fusion_checkpoint(
            fusion_path, fusion_config_for_variant("no_edc"), device
        )

    assigned = split_frame(frame, fold)
    if dataset == "ACAll":
        assigned["group"] = "ACAll"
    stable = _stable_for(dataset)
    rows = []
    aep = aep_config or {"sensitivity": 2.5, "min_size": 8, "max_points": 15, "max_tokens": 16}
    for unit, unit_frame in assigned.groupby("unit", sort=True):
        unit_frame = unit_frame.sort_values("cycle")
        full_di = unit_frame.DI.to_numpy(np.float32)
        group = str(unit_frame.group.iloc[0])
        role = str(unit_frame.split.iloc[0])
        for end in endpoints(len(unit_frame), **(prefix_config or {})):
            observed = unit_frame.iloc[:end]
            di = full_di[:end]
            di, dwc, edc, candidates, stable_fraction, reliability = _prefix_components(
                di, variant, fusion_model, stable, device
            )
            width = 128
            if variant == "no_decomposition":
                edc_predictor = np.zeros_like(edc)
                tokens, token_mask, _ = event_tokens(di, **aep)
            elif variant == "no_edc":
                edc_predictor = np.zeros_like(edc)
                tokens = np.zeros((16, 12), dtype=np.float32)
                token_mask = np.zeros(16, dtype=bool)
            elif variant == "no_aep":
                edc_predictor = edc
                tokens, token_mask = global_event_token(edc, aep["max_tokens"])
            else:
                edc_predictor = edc
                tokens, token_mask, _ = event_tokens(edc, **aep)
            dwc_input = dwc
            rows.append({
                "unit": int(unit), "group": group, "split": role,
                "endpoint": float(observed.cycle.iloc[-1]),
                "prefix_length": int(end),
                "true_rul_cycles": float(observed.RUL.iloc[-1]),
                "di": resize(di, width), "dwc": resize(dwc_input, width),
                "edc": resize(edc_predictor, width),
                "trend_candidates": np.stack((resize(candidates[0], width), resize(candidates[1], width))),
                "stable_fraction": np.float32(stable_fraction),
                "wiener_reliability": np.float32(reliability),
                "tokens": tokens.astype(np.float32), "token_mask": token_mask.astype(bool),
                "cache_signature": signature_key,
            })
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cache_version": 3, "cache_signature": signature, "rows": rows}, cache)
    return rows, fusion_path, cache
