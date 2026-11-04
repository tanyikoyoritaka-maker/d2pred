#!/usr/bin/env python3
"""Run the registered one-factor sensitivity grid after v2 comparisons."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_ac_nested_gru_rerun.py"
GROUPS = ROOT / "manifests/group_manifest.csv"
FOLDS = ROOT / "manifests/folds.json"
SEEDS = "17,42,73"
FACTORS = {
    "gamma": ("aep-sensitivity", [1.5, 2.0, 2.5, 3.0, 3.5]),
    "min_size": ("aep-min-size", [4, 8, 12, 16]),
    "max_points": ("aep-max-points", [8, 15, 24]),
    "max_tokens": ("aep-max-tokens", [8, 12, 16, 24, 32]),
    "prefix_start": ("prefix-start", [10, 30, 50]),
    "prefix_stride": ("prefix-stride", [10, 20, 40]),
    "max_prefixes": ("max-prefixes", [10, 20, 40]),
}
DEFAULTS = {k: 2.5 if k == "gamma" else 8 if k == "min_size" else 15 if k == "max_points" else 16 if k == "max_tokens" else 30 if k == "prefix_start" else 20 if k == "prefix_stride" else 20 for k in FACTORS}

def now(): return datetime.now(timezone.utc).isoformat()
def label(v): return str(v).replace('.', 'p')
def metrics(pred):
    keys = ["outer_fold", "seed", "model", "group", "unit"]
    rows=[]
    for vals, part in pred.groupby(keys, sort=True):
        e=part.error_cycles.to_numpy(float); a=np.abs(e)
        rows.append(dict(zip(keys, vals), num_predictions=len(part), RMSE_cycles=float(np.sqrt(np.mean(e**2))), MAE_cycles=float(np.mean(a)), P90AE_cycles=float(np.percentile(a,90)), bias_cycles=float(np.mean(e))))
    dev=pd.DataFrame(rows); out=[]
    for (model, group), part in dev.groupby(["model","group"], sort=True):
        dm=part.groupby("unit", as_index=False)[["RMSE_cycles","MAE_cycles","P90AE_cycles","bias_cycles"]].mean()
        rng=np.random.default_rng(20260915); vals=dm.RMSE_cycles.to_numpy(float)
        draws=np.array([rng.choice(vals,len(vals),replace=True).mean() for _ in range(2000)])
        out.append(dict(model=model,group=group,num_test_devices=len(dm),device_mean_RMSE_cycles=float(dm.RMSE_cycles.mean()),RMSE_95CI_low=float(np.percentile(draws,2.5)),RMSE_95CI_high=float(np.percentile(draws,97.5)),device_mean_MAE_cycles=float(dm.MAE_cycles.mean()),device_mean_P90AE_cycles=float(dm.P90AE_cycles.mean()),device_mean_bias_cycles=float(dm.bias_cycles.mean())))
    return dev,pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root-output", type=Path, required=True); args=ap.parse_args()
    out=args.root_output; out.mkdir(parents=True, exist_ok=True); records=[]
    manifest={"status":"running","started":now(),"shared_decomposition":False,"datasets":["ACSim","ACCom","ACAll"],"factors":FACTORS,"defaults":DEFAULTS}
    (out/"execution_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    for dataset in ["ACSim","ACCom","ACAll"]:
        src=out.parent/"comparison"/dataset/"test_predictions.csv"
        p=pd.read_csv(src); p=p[p.model.eq("d2pred_expert_exact_v1")].copy()
        default=out/f"{dataset}_default"; default.mkdir(parents=True,exist_ok=True)
        d,s=metrics(p); p.to_csv(default/"test_predictions.csv",index=False); d.to_csv(default/"device_level_metrics.csv",index=False); s.to_csv(default/"summary_metrics.csv",index=False)
        (default/"run_status.json").write_text(json.dumps({"status":"verified_complete","source":str(src),"default_point_from_v2_comparison":True},indent=2)+"\n")
        for factor,(flag,values) in FACTORS.items():
            for value in values:
                case=default if value==DEFAULTS[factor] else out/f"{dataset}_{factor}_{label(value)}"
                if value!=DEFAULTS[factor]:
                    status_file = case / "run_status.json"
                    if status_file.exists():
                        try:
                            if json.loads(status_file.read_text()).get("status") == "verified_complete":
                                print(f"[skip verified] {dataset} {factor}={value}", flush=True)
                                summ=pd.read_csv(case/"summary_metrics.csv")
                                for _, row in summ.iterrows(): records.append({"dataset":dataset,"factor":factor,"value":value,"default":False,**row.to_dict()})
                                continue
                        except (OSError, json.JSONDecodeError, FileNotFoundError):
                            pass
                    cmd=[sys.executable,"-u",str(RUNNER),"--output",str(case),"--profile","core","--models","d2pred_expert_exact_v1","--outer-folds","3","--seeds",SEEDS,"--epochs","50","--fusion-epochs","50","--batch-size","8","--device","cuda","--group-manifest",str(GROUPS),"--folds-file",str(FOLDS)]
                    cmd += ["--mix-groups"] if dataset=="ACAll" else ["--only-group",dataset]
                    cmd += [f"--{flag}",str(value)]; case.mkdir(parents=True,exist_ok=True)
                    (case/"command_v2.json").write_text(json.dumps({"command":cmd,"dataset":dataset,"factor":factor,"value":value},indent=2)+"\n")
                    subprocess.run(cmd,cwd=ROOT,check=True)
                summ=pd.read_csv(case/"summary_metrics.csv")
                for _, row in summ.iterrows(): records.append({"dataset":dataset,"factor":factor,"value":value,"default":value==DEFAULTS[factor],**row.to_dict()})
    allm=pd.DataFrame(records); allm.to_csv(out/"sensitivity_summary_all_metrics.csv",index=False)
    for dataset in ["ACSim","ACCom","ACAll"]: allm[allm.dataset.eq(dataset)].to_csv(out/f"sensitivity_{dataset}.csv",index=False)
    manifest.update(status="verified_complete",completed=now(),num_nondefault_runs=57,num_rows=len(allm)); (out/"execution_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    rows=[]
    for f in sorted(out.rglob("*")):
        if f.is_file() and f.name!="checksum_manifest.csv": rows.append({"path":str(f.relative_to(out)),"sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"bytes":f.stat().st_size})
    pd.DataFrame(rows).to_csv(out/"checksum_manifest.csv",index=False)

if __name__ == "__main__": main()
