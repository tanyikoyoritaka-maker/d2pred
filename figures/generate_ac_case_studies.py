"""Generate frozen AC decomposition and RUL case-study figures.

All rows are read from the frozen AC caches/prediction-level files; no training is run.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/ac_case_studies_frozen"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "ACSim": ROOT / "results/final_ac_nested",
    "ACCom": ROOT / "results/final_ac_nested",
    "ACAll": ROOT / "results/acall_nested",
}

def cache_rows(path):
    rows = []
    for f in sorted((path / "cache").glob("outer_fold_*_rows.pt")):
        payload = torch.load(f, map_location="cpu", weights_only=False)
        rows.extend(payload["rows"])
    return rows

def choose_unit(group, rows):
    # Keep the previously registered ACSim/ACCom examples; use the same
    # anonymized unit for the pooled ACAll view when it is available.
    registered = {"ACSim": 10, "ACCom": 12, "ACAll": 12}
    test = [r for r in rows if r["split"] == "test" and (group == "ACAll" or r["group"] == group)]
    units = sorted({int(r["unit"]) for r in test})
    if not units:
        raise RuntimeError(f"No test unit found for {group}")
    if registered[group] in units:
        return registered[group]
    return units[len(units) // 2]

def final_row(rows, unit, group):
    candidates = [r for r in rows if r["split"] == "test" and int(r["unit"]) == unit]
    if group != "ACAll":
        candidates = [r for r in candidates if r["group"] == group]
    return sorted(candidates, key=lambda r: (float(r["endpoint"]), int(r["prefix_length"])))[-1]

def make_decomposition():
    chosen, records = {}, []
    for group, source in SOURCES.items():
        rows = cache_rows(source)
        unit = choose_unit(group, rows)
        row = final_row(rows, unit, group)
        chosen[group] = unit
        x = np.linspace(0.0, 1.0, len(row["di"]))
        for name in ("edc", "dwc", "di"):
            records.append({"group": group, "unit": unit, "component": name, "x": x, "value": row[name]})
    pd.DataFrame([{"group": g, "unit": u} for g, u in chosen.items()]).to_csv(OUT / "selected_units.csv", index=False)
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 10.0), sharex=True)
    colors = {"edc": "#2ca02c", "dwc": "#ff7f0e", "di": "#1f77b4"}
    labels = {"edc": "EDC", "dwc": "DWC", "di": "DI"}
    groups = ["ACSim", "ACCom", "ACAll"]
    for i, group in enumerate(groups):
        row = final_row(cache_rows(SOURCES[group]), chosen[group], group)
        for j, name in enumerate(("edc", "dwc", "di")):
            ax = axes[i, j]
            ax.plot(np.linspace(0, 1, len(row[name])), row[name], color=colors[name], lw=1.25)
            ax.set_title(f"({chr(97+i*3+j)}) {group}: {labels[name]}", fontsize=10, fontweight="bold")
            ax.grid(alpha=.3, lw=.5)
            ax.set_xlim(0, 1)
            if i == 2: ax.set_xlabel("Operation cycle")
            if j == 0: ax.set_ylabel("Value")
        axes[i, 0].text(1.03, .5, "+", transform=axes[i, 0].transAxes, fontsize=22, va="center", ha="center")
        axes[i, 1].text(1.03, .5, "=", transform=axes[i, 1].transAxes, fontsize=22, va="center", ha="center")
    pd.DataFrame(records).explode(["x", "value"]).to_csv(OUT / "ac_di_decomposition_source_data.csv", index=False)
    fig.suptitle("Frozen AC decomposition examples", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(OUT / "ac_di_decomposition_frozen.pdf", bbox_inches="tight")
    fig.savefig(OUT / "ac_di_decomposition_frozen.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return chosen

def make_rul(chosen):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.3), sharex=True, sharey=True)
    all_records = []
    plotted = []
    for group_index, (ax, group) in enumerate(zip(axes, ("ACSim", "ACCom", "ACAll"))):
        source = SOURCES[group]
        pred = pd.read_csv(source / "seed_ensemble_predictions_with_intervals.csv")
        pred = pred[(pred.model == "d2pred_expert_exact_v1") & (pred.unit == chosen[group])].copy()
        pred = pred.sort_values(["outer_fold", "endpoint"]).drop_duplicates(["endpoint"], keep="first")
        x = (pred.endpoint - pred.endpoint.min()) / max(pred.endpoint.max() - pred.endpoint.min(), 1.0)
        initial = float(pred.true_rul_cycles.iloc[0])
        true_line = 1.0 - x
        prediction = pred.ensemble_prediction.to_numpy(float) / max(initial, 1.0)
        vertical_pct = 100.0 * (prediction - true_line)
        all_records.extend(pd.DataFrame({"group": group, "unit": chosen[group], "x": x,
            "true_rul_normalized": true_line, "predicted_rul_normalized": prediction,
            "vertical_distance_percent": vertical_pct}).to_dict("records"))
        ax.plot(x, true_line, color="#1f77b4", lw=1.8, label="Original RUL (slope -1)")
        ax.plot(x, prediction, color="#ff7f0e", lw=1.6, ls="--", label="Predicted RUL")
        plotted.append(prediction)
        ax.set_title(f"({chr(97 + group_index)}) {group}", fontweight="bold")
        ax.set_xlabel("Normalized operation cycle")
        ax.set_ylabel("Normalized RUL")
        ax.set_xlim(0, 1); ax.grid(alpha=.3)
    ymax = max(1.05, max(float(np.nanmax(v)) for v in plotted) * 1.05)
    ymin = min(-0.05, min(float(np.nanmin(v)) for v in plotted) - 0.05)
    for ax in axes:
        ax.set_ylim(ymin, ymax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(.5, .98), frameon=False)
    fig.suptitle("RUL prediction on frozen AC trajectories", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "ac_rul_prediction_frozen.pdf", bbox_inches="tight")
    fig.savefig(OUT / "ac_rul_prediction_frozen.png", dpi=300, bbox_inches="tight")
    pd.DataFrame(all_records).to_csv(OUT / "ac_rul_prediction_source_data.csv", index=False)
    plt.close(fig)

if __name__ == "__main__":
    chosen = make_decomposition()
    make_rul(chosen)
    print(chosen)
