"""Generate all ACAll frozen ablation RUL case-study comparisons."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'results/acall_preprocess_ablation_final/seed_ensemble_predictions_with_intervals.csv'
OUT = ROOT / 'figures/acall_ablation_rul_cases'
OUT.mkdir(parents=True, exist_ok=True)
VARIANTS = {
    'd2pred_expert_exact_v1': 'Full model',
    'ablation_preprocess_no_edc': 'Without EDC',
    'ablation_preprocess_no_decomposition': 'Without decomposition',
}

def normalized_case(df):
    df = df.sort_values('endpoint').drop_duplicates('endpoint', keep='first')
    x = (df.endpoint - df.endpoint.min()) / max(df.endpoint.max() - df.endpoint.min(), 1.0)
    initial = float(df.true_rul_cycles.iloc[0])
    true = 1.0 - x.to_numpy(float)
    pred = df.ensemble_prediction.to_numpy(float) / max(initial, 1.0)
    return x.to_numpy(float), true, pred

def main():
    data = pd.read_csv(SRC)
    data = data[data.model.isin(VARIANTS)].copy()
    units = sorted(data.unit.unique())
    rows = []
    with PdfPages(OUT / 'acall_without_edc_without_decomposition_rul_cases.pdf') as pdf:
        for unit in units:
            fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharex=True, sharey=True)
            for ax, (variant, label) in zip(axes, VARIANTS.items()):
                p = data[(data.unit == unit) & (data.model == variant)]
                if p.empty:
                    ax.axis('off'); continue
                x, true, pred = normalized_case(p)
                ax.plot(x, true, color='#1f77b4', lw=1.8, label='Original RUL (slope -1)')
                ax.plot(x, pred, color='#ff7f0e', lw=1.5, ls='--', label='Predicted RUL')
                ax.set_title(label, fontweight='bold')
                ax.set_xlabel('Normalized operation cycle')
                ax.set_ylabel('Normalized RUL')
                ax.set_xlim(0, 1); ax.grid(alpha=.3)
                rows.extend({'unit': int(unit), 'variant': label, 'x': float(xx),
                             'original_rul_normalized': float(tt),
                             'predicted_rul_normalized': float(pp),
                             'vertical_distance_percent': float(100*(pp-tt))}
                            for xx, tt, pp in zip(x, true, pred))
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
            fig.suptitle(f'ACAll frozen ablation RUL case: anonymized Unit {int(unit)}', y=.99, fontweight='bold')
            fig.tight_layout(rect=(0, 0, 1, .92))
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / 'acall_ablation_rul_case_source_data.csv', index=False)
    # Contact sheet of all devices, with each column showing one ablation.
    n = len(units); fig, axes = plt.subplots(n, 2, figsize=(10, max(2.0*n, 8)), squeeze=False, sharex=True, sharey=True)
    for i, unit in enumerate(units):
        for j, (variant, label) in enumerate(VARIANTS.items()):
            ax = axes[i, j]; p = data[(data.unit == unit) & (data.model == variant)]
            if p.empty: ax.axis('off'); continue
            x, true, pred = normalized_case(p)
            ax.plot(x, true, color='#1f77b4', lw=.7); ax.plot(x, pred, color='#ff7f0e', ls='--', lw=.7)
            ax.set_title(f'Unit {int(unit)}: {label}', fontsize=8); ax.grid(alpha=.2); ax.set_xlim(0, 1)
    fig.suptitle('All ACAll frozen ablation RUL cases', y=.998, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .995))
    fig.savefig(OUT / 'acall_ablation_rul_cases_contact_sheet.png', dpi=220, bbox_inches='tight'); plt.close(fig)
    print({'units': len(units), 'variants': list(VARIANTS.values()), 'pdf': str(OUT / 'acall_without_edc_without_decomposition_rul_cases.pdf')})

if __name__ == '__main__': main()
