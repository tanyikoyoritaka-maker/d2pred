"""Generate all frozen test-device DI decomposition case studies.

The model has no separate DI decoder. Therefore the plotted predicted DI is
the exact causal reconstruction DI_hat = DWC + EDC from the frozen cache.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures/all_frozen_di_cases"
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {
    "ACSim": ROOT / "results/final_ac_nested",
    "ACCom": ROOT / "results/final_ac_nested",
    "ACAll": ROOT / "results/acall_nested",
}

def load_rows(source):
    rows = []
    for f in sorted((source / "cache").glob("outer_fold_*_rows.pt")):
        rows.extend(torch.load(f, map_location="cpu", weights_only=False)["rows"])
    return rows

def final_test_rows(group, rows):
    data = [r for r in rows if r["split"] == "test" and (group == "ACAll" or r["group"] == group)]
    selected = {}
    for r in data:
        unit = int(r["unit"])
        key = (float(r["endpoint"]), int(r["prefix_length"]))
        if unit not in selected or key > selected[unit][0]:
            selected[unit] = (key, r)
    return [selected[u][1] for u in sorted(selected)]

def one_page(group, row):
    di = np.asarray(row["di"], dtype=float)
    dwc = np.asarray(row["dwc"], dtype=float)
    edc = np.asarray(row["edc"], dtype=float)
    reconstructed = dwc + edc
    x = np.linspace(0, 1, len(di))
    err = reconstructed - di
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    ax = axes[0, 0]
    ax.plot(x, di, color="#1f77b4", lw=1.5, label="Original DI")
    ax.plot(x, reconstructed, color="#d62728", lw=1.2, ls="--", label=r"Reconstructed $\widehat{DI}=DWC+EDC$")
    ax.set_title("Original DI vs reconstructed DI", fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax = axes[0, 1]
    ax.plot(x, dwc, color="#ff7f0e", lw=1.4)
    ax.set_title("DWC: gradual component", fontweight="bold")
    ax = axes[1, 0]
    ax.plot(x, edc, color="#2ca02c", lw=1.4)
    ax.set_title("EDC: residual component", fontweight="bold")
    ax = axes[1, 1]
    ax.plot(x, err, color="#9467bd", lw=1.3)
    ax.axhline(0, color="black", lw=.8)
    ax.set_title(f"Reconstruction error (max |e|={np.max(np.abs(err)):.2e})", fontweight="bold")
    for ax in axes.ravel():
        ax.grid(alpha=.3, lw=.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Normalized operation cycle")
        ax.set_ylabel("DI value")
    fig.suptitle(f"Frozen AC case study: {group}, anonymized unit {int(row['unit'])}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .95))
    return fig, {
        "group": group, "unit": int(row["unit"]), "endpoint": float(row["endpoint"]),
        "max_abs_reconstruction_error": float(np.max(np.abs(err))),
        "rmse_reconstruction_error": float(np.sqrt(np.mean(err ** 2))),
    }

def main():
    summaries = []
    for group, source in SOURCES.items():
        rows = final_test_rows(group, load_rows(source))
        with PdfPages(OUT / f"{group.lower()}_all_frozen_di_cases.pdf") as pdf:
            for row in rows:
                fig, summary = one_page(group, row)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                summaries.append(summary)
        # A compact contact sheet is useful for rapid audit while the PDF
        # retains one full-resolution page per device.
        cols = 4
        rcount = int(np.ceil(len(rows) / cols))
        fig, axes = plt.subplots(rcount, cols, figsize=(14, 2.7 * rcount), squeeze=False)
        for i, row in enumerate(rows):
            ax = axes.ravel()[i]
            di = np.asarray(row["di"], float); rec = np.asarray(row["dwc"], float) + np.asarray(row["edc"], float)
            ax.plot(np.linspace(0, 1, len(di)), di, color="#1f77b4", lw=.8, label="DI")
            ax.plot(np.linspace(0, 1, len(rec)), rec, color="#d62728", lw=.7, ls="--", label="DI_hat")
            ax.set_title(f"Unit {int(row['unit'])}", fontsize=9); ax.grid(alpha=.25); ax.set_xlim(0, 1)
        for ax in axes.ravel()[len(rows):]: ax.axis("off")
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.suptitle(f"All frozen {group} DI cases", y=.995, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, .98))
        fig.savefig(OUT / f"{group.lower()}_all_frozen_di_cases_contact_sheet.png", dpi=250, bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame(summaries).to_csv(OUT / "all_frozen_di_case_reconstruction_metrics.csv", index=False)
    print(pd.DataFrame(summaries).groupby("group").size().to_dict())

if __name__ == "__main__":
    main()
