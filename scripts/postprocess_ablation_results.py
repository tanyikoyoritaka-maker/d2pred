#!/usr/bin/env python3
"""Complete calibration and paired tests from finished ablation predictions."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SETS = ["acsim_preprocess_ablation_final", "accom_preprocess_ablation_final", "acall_preprocess_ablation_final"]
REFERENCE = "d2pred_expert_exact_v1"

def postprocess(out: Path) -> dict:
    val = pd.read_csv(out / "validation_predictions.csv")
    test = pd.read_csv(out / "test_predictions.csv")
    calibration = []
    for (fold, model, group), part in val.groupby(["outer_fold", "model", "group"], sort=True):
        radius = float(np.quantile(np.abs(part.error_cycles.to_numpy(float)), .95))
        calibration.append({"outer_fold": int(fold), "model": model, "group": group,
                           "calibration_radius_cycles": radius,
                           "quantile": .95, "num_validation_predictions": len(part),
                           "fit_split": "inner_validation_only"})
    cal = pd.DataFrame(calibration)
    cal.to_csv(out / "validation_calibration.csv", index=False)
    cal_key = ["outer_fold", "model", "group"]
    test_ensemble = test.groupby(["outer_fold", "model", "unit", "group", "endpoint"], as_index=False).agg(
        true_rul_cycles=("true_rul_cycles", "first"),
        ensemble_prediction=("predicted_rul_cycles", "mean"),
        seed_std=("predicted_rul_cycles", "std"),
    )
    test_ensemble = test_ensemble.merge(cal[cal_key + ["calibration_radius_cycles"]], on=cal_key, how="left", validate="many_to_one")
    test_ensemble["seed_std"] = test_ensemble["seed_std"].fillna(0.0)
    test_ensemble["interval_low"] = (test_ensemble.ensemble_prediction - test_ensemble.calibration_radius_cycles).clip(lower=0)
    test_ensemble["interval_high"] = test_ensemble.ensemble_prediction + test_ensemble.calibration_radius_cycles
    test_ensemble["covered"] = ((test_ensemble.true_rul_cycles >= test_ensemble.interval_low) &
                                 (test_ensemble.true_rul_cycles <= test_ensemble.interval_high))
    test_ensemble.to_csv(out / "seed_ensemble_predictions_with_intervals.csv", index=False)
    reliability = test_ensemble.groupby(["model", "group"], as_index=False).agg(
        empirical_coverage=("covered", "mean"),
        mean_interval_width=("calibration_radius_cycles", lambda x: float(2*x.mean())),
        mean_seed_std=("seed_std", "mean"), num_predictions=("covered", "size"))
    reliability.to_csv(out / "reliability_summary.csv", index=False)

    unit = []
    for keys, part in test.groupby(["outer_fold", "seed", "model", "group", "unit"], sort=True):
        err = part.error_cycles.to_numpy(float)
        unit.append(dict(zip(["outer_fold", "seed", "model", "group", "unit"], keys)) |
                     {"RMSE_cycles": float(np.sqrt(np.mean(err**2)))})
    unit = pd.DataFrame(unit)
    records = []
    for group, part in unit.groupby("group", sort=True):
        pivot = part.groupby(["unit", "model"], as_index=False).RMSE_cycles.mean().pivot(index="unit", columns="model", values="RMSE_cycles")
        if REFERENCE not in pivot.columns:
            continue
        for model in pivot.columns:
            if model == REFERENCE:
                continue
            pair = pivot[[REFERENCE, model]].dropna()
            diff = pair[model] - pair[REFERENCE]
            try: stat, p = wilcoxon(diff)
            except ValueError: stat, p = 0.0, 1.0
            records.append({"group": group, "reference": REFERENCE, "comparison": model,
                            "num_paired_devices": len(pair),
                            "mean_RMSE_difference_comparison_minus_reference": float(diff.mean()),
                            "wilcoxon_statistic": float(stat), "p_value": float(p),
                            "paired_cohen_dz": float(diff.mean()/diff.std(ddof=1)) if len(diff)>1 and diff.std(ddof=1)>0 else 0.0})
    tests = pd.DataFrame(records)
    if len(tests):
        tests["holm_threshold_rank"] = tests.p_value.rank(method="first").astype(int)
        tests["holm_significant_0.05"] = tests.p_value <= 0.05 / (len(tests)-tests.holm_threshold_rank+1)
    tests.to_csv(out / "paired_statistical_tests.csv", index=False)
    files=[]
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "checksum_manifest.csv":
            files.append({"path":str(path.relative_to(out)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size})
    pd.DataFrame(files).to_csv(out / "checksum_manifest.csv", index=False)
    numeric = test_ensemble.select_dtypes(include=[np.number]).to_numpy()
    return {"calibration_rows":len(cal),"ensemble_rows":len(test_ensemble),"paired_rows":len(tests),
            "nonfinite":int((~np.isfinite(numeric)).sum())}

if __name__ == "__main__":
    print(json.dumps({name: postprocess(ROOT / "results" / name) for name in SETS}, indent=2))
