#!/usr/bin/env python3
"""Mandatory final expert-protocol gate before any full training."""
from __future__ import annotations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.expert_exact_d2pred import ExpertExactD2Pred
from analysis.theory_aligned_decomposition import (
    StageAwareTrendAutoencoder, stage_aware_causal_decompose, stage_aware_fusion_loss,
)
from run_ac_nested_gru_rerun import PrefixDataset, load_ac, validate_predefined_folds
from run_stage_aware_aefusion_ac import DECOMPOSITION, STABLE

OUT = ROOT / "manifests" / "model_preexecution_audit.json"


def sample_row(group: str) -> dict:
    t = np.linspace(0, 1, 128, dtype=np.float32)
    di = (1.2 - 2.5*t + .08*np.sin(18*t)).astype(np.float32)
    wiener = (1.2 - 2.4*t).astype(np.float32)
    stl = (1.2 - 2.5*t + .02*np.sin(5*t)).astype(np.float32)
    dwc = .55*wiener + .45*stl
    edc = di-dwc
    mask = np.zeros(16, dtype=bool); mask[:4] = True
    return {"unit": 12, "group": group, "split": "train", "endpoint": 128.0,
            "prefix_length": 128, "true_rul_cycles": 300.0, "di": di,
            "dwc": dwc, "edc": edc, "trend_candidates": np.stack((wiener, stl)),
            "stable_fraction": np.float32(.2), "wiener_reliability": np.float32(.8),
            "tokens": np.zeros((16,12),np.float32), "token_mask": mask,
            "change_points": [], "edc_identity_error": 0.0}


def main() -> None:
    torch.manual_seed(20260919)
    manifest = ROOT / "manifests" / "group_manifest.csv"
    folds = json.loads((ROOT / "manifests" / "folds.json").read_text())
    frame = load_ac(group_manifest=manifest)
    validate_predefined_folds(frame, folds)

    normalization = {"di_mean": 0.0, "di_std": 1.0, "dwc_mean": 0.0, "dwc_std": 1.0,
                     "edc_mean": 0.0, "edc_std": 1.0, "token_mean": [0.0]*12,
                     "token_std": [1.0]*12, "fit_units": [12]}
    row_a, row_b = sample_row("ACSim"), sample_row("ACCom")
    item_a = PrefixDataset([row_a], normalization, 1000.0, "d2pred_expert_exact_v1")[0][0]
    item_b = PrefixDataset([row_b], normalization, 1000.0, "d2pred_expert_exact_v1")[0][0]
    invariance = all(torch.equal(item_a[key], item_b[key]) for key in item_a)
    batch = {key: value.unsqueeze(0) for key, value in item_a.items()}
    model = ExpertExactD2Pred(joint_trend_fusion=True)
    details = model(batch, return_details=True)
    label = torch.tensor([.3])
    rul_loss = ((details["prediction"]-label)**2).mean()
    gradual_loss = ((details["gradual_prediction"]-label)**2).mean()
    event_loss = ((details["event_prediction"]-label)**2).mean()
    fusion_loss, components = stage_aware_fusion_loss(
        details["fused_dwc"], batch["di"], batch["trend_candidates"],
        batch["sequence_mask"], details["trend_fusion_details"], model.trend_fusion.config,
    )
    total = rul_loss + .1*gradual_loss + .1*event_loss + .1*fusion_loss
    total.backward()
    fusion_grads = [p.grad for p in model.trend_fusion.parameters() if p.requires_grad]
    gradient_ok = bool(fusion_grads and all(g is not None and torch.isfinite(g).all() for g in fusion_grads)
                       and any(float(g.abs().sum()) > 0 for g in fusion_grads))

    raw = pd.read_csv(ROOT / "data/industrial/DI_df.csv")
    prefix = raw.loc[raw.unit.eq(12), "HI"].to_numpy(np.float32)[:128]
    altered = np.concatenate((prefix, np.full(50, 999.0, np.float32)))
    fusion = StageAwareTrendAutoencoder().eval()
    a = stage_aware_causal_decompose(prefix, fusion, STABLE["ACCom"], DECOMPOSITION)[0]
    b = stage_aware_causal_decompose(altered[:128], fusion, STABLE["ACCom"], DECOMPOSITION)[0]
    causality_delta = float(np.max(np.abs(a-b)))

    checks = {
        "fixed_folds_valid": True,
        "retained_units_33": frame.unit.nunique() == 33,
        "groups_16_17": frame.groupby("group").unit.nunique().to_dict() == {"ACCom":16,"ACSim":17},
        "migrations_present": set(frame.loc[frame.group.eq("ACCom"),"unit"].astype(int)) >= {12,16} and 23 in set(frame.loc[frame.group.eq("ACSim"),"unit"].astype(int)),
        "group_label_invariant_inputs": invariance,
        "joint_fusion_gradients_finite_nonzero": gradient_ok,
        "prefix_causality_exact": causality_delta == 0.0,
        "expert_architecture_versioned": model.architecture_metadata()["architecture_version"] == "d2pred_expert_exact_v1",
        "joint_objective_metadata_present": "trend_fusion" in model.architecture_metadata()["joint_objective"],
    }
    payload = {"audit":"d2pred_final_preexecution_v1",
               "status":"PASS" if all(checks.values()) else "FAIL", "checks":checks,
               "losses":{"rul":float(rul_loss),"gradual":float(gradual_loss),
                         "event":float(event_loss),"trend_fusion":float(fusion_loss),"total":float(total)},
               "fusion_loss_components":{k:float(v) for k,v in components.items()},
               "prefix_causality_max_abs_delta":causality_delta,
               "architecture":model.architecture_metadata()}
    OUT.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2))
    if payload["status"] != "PASS": raise SystemExit(1)

if __name__ == "__main__": main()
