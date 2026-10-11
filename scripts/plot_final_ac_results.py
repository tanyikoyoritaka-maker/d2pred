"""Create the verified AC final-results table and bar-chart report."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "final_ac_nested"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(RESULTS / "summary_metrics.csv")
df["model"] = df["model"].replace({"d2pred_expert_exact_v1": "D2Pred Expert"})
order = ["ACSim", "ACCom"]
df["group"] = pd.Categorical(df["group"], categories=order, ordered=True)
df = df.sort_values(["group", "device_mean_RMSE_cycles", "model"]).reset_index(drop=True)

metrics = [
    ("device_mean_RMSE_cycles", "RMSE (cycles)"),
    ("device_mean_MAE_cycles", "MAE (cycles)"),
    ("device_mean_P90AE_cycles", "P90AE (cycles)"),
]
colors = {"ACSim": "#4472C4", "ACCom": "#ED7D31"}

def chart(ax, col, title):
    models = sorted(df["model"].unique())
    x = np.arange(len(models))
    width = 0.38
    for i, group in enumerate(order):
        sub = df[df.group == group].set_index("model").reindex(models)
        vals = sub[col].to_numpy(float)
        err = None
        if col == "device_mean_RMSE_cycles":
            low = vals - sub["RMSE_95CI_low"].to_numpy(float)
            high = sub["RMSE_95CI_high"].to_numpy(float) - vals
            err = np.vstack([low, high])
        ax.bar(x + (i - .5) * width, vals, width, label=group,
               color=colors[group], alpha=.9, yerr=err, capsize=2,
               error_kw={"elinewidth": .7})
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("cycles")
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False, ncol=2)

pdf_path = OUT / "final_ac_results_report.pdf"
png_path = OUT / "final_ac_results_dashboard.png"

with PdfPages(pdf_path) as pdf:
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Final AC nested device-wise evaluation", fontsize=18, weight="bold")
    fig.text(.5, .94, "Verified outputs: 3 outer folds × seeds {17, 42, 73} × 11 models | lower is better",
             ha="center", fontsize=10)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], hspace=.45, wspace=.25)
    for ax, (col, title) in zip([fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])], metrics):
        chart(ax, col, title)
    fig.text(.01, .01, "RMSE bars show bootstrap device-level 95% CIs. Source: results/final_ac_nested/summary_metrics.csv",
             fontsize=8, color="dimgray")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # Reliability and computational results are part of the same verified run.
    rel = pd.read_csv(RESULTS / "reliability_summary.csv")
    rel["model"] = rel["model"].replace({"d2pred_expert_exact_v1": "D2Pred Expert"})
    cost = pd.read_csv(RESULTS / "computational_cost.csv")
    cost["model"] = cost["model"].replace({"d2pred_expert_exact_v1": "D2Pred Expert"})
    cagg = cost.groupby("model", as_index=False).agg(
        training_seconds=("training_seconds", "median"),
        parameter_count=("parameter_count", "first"))
    models = sorted(df["model"].unique())
    x = np.arange(len(models)); width = .38
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for i, group in enumerate(order):
        s = rel[rel.group == group].set_index("model").reindex(models)
        axes[0, 0].bar(x + (i-.5)*width, s["empirical_coverage"].to_numpy()*100, width, label=group, color=colors[group])
        axes[0, 1].bar(x + (i-.5)*width, s["mean_interval_width"].to_numpy(), width, label=group, color=colors[group])
    axes[0, 0].axhline(95, color="black", ls="--", lw=1, label="95% target")
    axes[0, 0].set_title("Empirical interval coverage (%)", weight="bold")
    axes[0, 1].set_title("Mean calibrated interval width (cycles)", weight="bold")
    for ax in axes[0]:
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7); ax.grid(axis="y", alpha=.25); ax.legend(frameon=False)
    axes[1, 0].bar(x, cagg.set_index("model").reindex(models)["training_seconds"], color="#70AD47")
    axes[1, 0].set_title("Median training time per run (s)", weight="bold")
    axes[1, 1].bar(x, cagg.set_index("model").reindex(models)["parameter_count"], color="#A5A5A5")
    axes[1, 1].set_title("Parameter count", weight="bold")
    for ax in axes[1]:
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7); ax.grid(axis="y", alpha=.25)
    fig.suptitle("Reliability and computational results", fontsize=16, weight="bold")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # Numeric table page, directly from the verified summary file.
    table_df = df[["model", "group", "num_test_devices", "device_mean_RMSE_cycles",
                   "RMSE_95CI_low", "RMSE_95CI_high", "device_mean_MAE_cycles",
                   "device_mean_P90AE_cycles"]].copy()
    for c in table_df.columns[3:]:
        table_df[c] = table_df[c].map(lambda x: f"{x:.2f}")
    table_df.columns = ["Model", "Group", "N devices", "RMSE", "RMSE CI low", "RMSE CI high", "MAE", "P90AE"]
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis("off")
    ax.set_title("Complete verified results table (cycles; lower is better)", fontsize=16, weight="bold", pad=18)
    tbl = ax.table(cellText=table_df.values, colLabels=table_df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.55)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1F4E78"); cell.set_text_props(color="white", weight="bold")
        elif r > 0 and table_df.iloc[r-1, 1] == "ACCom":
            cell.set_facecolor("#FCE4D6")
        elif r > 0:
            cell.set_facecolor("#D9EAF7")
    ax.text(.01, .03, "Each row is a group-level aggregate over the held-out devices across the fixed nested evaluation."
            " RMSE interval columns are the reported device-level 95% intervals.", transform=ax.transAxes, fontsize=9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print(pdf_path)
print(png_path)
