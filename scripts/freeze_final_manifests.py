#!/usr/bin/env python3
"""Freeze leakage-safe AC cohort, endpoint, grouping, and fold manifests."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "PythonProject2"
EXCLUDED = {1, 3, 11, 35, 38}
EXPECTED = set(range(1, 39)) - EXCLUDED
BASELINES = ["CNN", "DAG_network", "PM2FN", "Transformer", "TCN_BiGRU",
             "KGHM", "ATCN", "MDFA", "SBi_Transformer", "Attn_ResDSCNN"]

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    manifests = ROOT / "manifests"; configs = ROOT / "configs"
    manifests.mkdir(parents=True, exist_ok=True); configs.mkdir(parents=True, exist_ok=True)
    di_path = SOURCE / "data/DI/DI_df.csv"
    groups_path = SOURCE / "exp_cycles_12_16_review/group_manifest.csv"
    folds_path = SOURCE / "exp_cycles_12_16_review/folds.json"
    di = pd.read_csv(di_path); di["unit"] = di.unit.astype(int)
    endpoints = di.sort_values(["unit", "time_cycles"]).groupby("unit").agg(
        first_cycle=("time_cycles", "first"), endpoint_cycle=("time_cycles", "last"),
        initial_rul=("RUL", "first"), endpoint_rul=("RUL", "last"), rows=("RUL", "size"),
    ).reset_index()
    endpoints["endpoint_rule"] = "final cycle where RUL=0"
    endpoints["eligible"] = ~endpoints.unit.isin(EXCLUDED)
    endpoints["excluded_reason"] = endpoints.unit.map(lambda u: "different_device_type_ultra_long_lifetime" if u in EXCLUDED else "")
    endpoint_ok = ((endpoints.first_cycle == 1) & (endpoints.endpoint_rul == 0) &
                   (endpoints.initial_rul == endpoints.rows - 1) &
                   (endpoints.endpoint_cycle == endpoints.rows)).all()
    endpoints.to_csv(manifests / "failure_endpoint_manifest.csv", index=False)

    groups = pd.read_csv(groups_path); groups["unit"] = groups.unit.astype(int)
    folds = json.loads(folds_path.read_text())
    fold_checks=[]; test_union=[]
    for f in folds:
        tr,va,te=map(set,(f["train_units"],f["validation_units"],f["test_units"]))
        test_union += list(te)
        fold_checks.append({"outer_fold":f["outer_fold"],"disjoint":not(tr&va or tr&te or va&te),
                            "covers_33":tr|va|te==EXPECTED,"train":len(tr),"validation":len(va),"test":len(te)})
    checks = {
        "source_units_38": set(endpoints.unit)==set(range(1,39)), "endpoint_identity_all_38": bool(endpoint_ok),
        "eligible_units_33": set(groups.unit)==EXPECTED and len(groups)==33,
        "excluded_units_exact": sorted(EXCLUDED)==[1,3,11,35,38],
        "group_counts_16_17": groups.group.value_counts().to_dict()=={"ACSim":17,"ACCom":16},
        "unit_12_16_accom": set(groups.loc[groups.unit.isin([12,16]),"group"])=={"ACCom"},
        "unit_23_acsim": groups.loc[groups.unit.eq(23),"group"].tolist()==["ACSim"],
        "three_outer_folds": len(folds)==3, "folds_disjoint_and_complete": all(x["disjoint"] and x["covers_33"] for x in fold_checks),
        "test_once_each_across_folds": sorted(test_union)==sorted(EXPECTED) and len(test_union)==33,
        "baseline_registry_exact": len(BASELINES)==10 and not ({"LSTM","BiLSTM","IMDSSN"}&set(BASELINES)),
    }
    if not all(checks.values()): raise SystemExit(json.dumps(checks,indent=2))
    groups.to_csv(manifests / "group_manifest.csv", index=False)
    (manifests / "folds.json").write_text(json.dumps(folds,indent=2)+"\n")
    registry={"expert":"d2pred_expert_exact_v1","public_baselines":BASELINES,
              "removed":["LSTM","BiLSTM","IMDSSN"],"adapted_label":"architecture-level reimplementation"}
    (configs / "model_registry.json").write_text(json.dumps(registry,indent=2)+"\n")
    audit={"status":"PASS","checks":checks,"fold_checks":fold_checks,"source_checksums":{
        "DI_df.csv":sha(di_path),"source_group_manifest.csv":sha(groups_path),"source_folds.json":sha(folds_path)}}
    (manifests / "freeze_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))

if __name__ == "__main__": main()
