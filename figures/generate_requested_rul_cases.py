"""Generate requested frozen RUL case-study comparisons."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    'ACAll': (ROOT / 'results/acall_preprocess_ablation_final', {
        'd2pred_expert_exact_v1': 'Full model',
        'ablation_preprocess_no_edc': 'Without EDC',
        'ablation_preprocess_no_decomposition': 'Without decomposition',
    }),
    'ACSim': (ROOT / 'results/acsim_preprocess_ablation_final', {
        'd2pred_expert_exact_v1': 'Full model',
    }),
    'ACCom': (ROOT / 'results/accom_preprocess_ablation_final', {
        'd2pred_expert_exact_v1': 'Full model',
    }),
}
OUT = ROOT / 'figures/requested_rul_cases'
OUT.mkdir(parents=True, exist_ok=True)

def normalized(df):
    df = df.sort_values('endpoint').drop_duplicates('endpoint', keep='first')
    x = (df.endpoint - df.endpoint.min()) / max(df.endpoint.max()-df.endpoint.min(), 1.0)
    initial = max(float(df.true_rul_cycles.iloc[0]), 1.0)
    true = 1.0 - x.to_numpy(float)
    pred = df.ensemble_prediction.to_numpy(float) / initial
    return x.to_numpy(float), true, pred

def main():
    records = []
    for dataset, (source, variants) in CONFIG.items():
        data = pd.read_csv(source / 'seed_ensemble_predictions_with_intervals.csv')
        data = data[data.model.isin(variants)].copy()
        units = sorted(data.unit.unique())
        with PdfPages(OUT / f'{dataset.lower()}_requested_rul_cases.pdf') as pdf:
            for unit in units:
                fig, axes = plt.subplots(1, len(variants), figsize=(6.2*len(variants), 4.8), sharex=True, sharey=True, squeeze=False)
                for ax, (model, label) in zip(axes.ravel(), variants.items()):
                    p = data[(data.unit == unit) & (data.model == model)]
                    x, true, pred = normalized(p)
                    ax.plot(x, true, color='#1f77b4', lw=1.8, label='Original RUL (slope -1)')
                    ax.plot(x, pred, color='#ff7f0e', lw=1.5, ls='--', label='Predicted RUL')
                    ax.set_title(label, fontweight='bold'); ax.set_xlabel('Normalized operation cycle'); ax.set_ylabel('Normalized RUL')
                    ax.set_xlim(0, 1); ax.grid(alpha=.3)
                    records.extend({'dataset':dataset,'unit':int(unit),'model':label,'x':float(xx),
                                    'original_rul_normalized':float(tt),'predicted_rul_normalized':float(pp),
                                    'vertical_distance_percent':float(100*(pp-tt))}
                                   for xx,tt,pp in zip(x,true,pred))
                handles, labels = axes.ravel()[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)
                fig.suptitle(f'{dataset} frozen RUL case: anonymized Unit {int(unit)}', y=.99, fontweight='bold')
                fig.tight_layout(rect=(0,0,1,.92)); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        # Contact sheet for all units in this dataset.
        cols = len(variants); rows = len(units)
        fig, axes = plt.subplots(rows, cols, figsize=(5.1*cols, max(2.0*rows, 6)), squeeze=False, sharex=True, sharey=True)
        for i, unit in enumerate(units):
            for j, (model, label) in enumerate(variants.items()):
                ax=axes[i,j]; p=data[(data.unit==unit)&(data.model==model)]; x,true,pred=normalized(p)
                ax.plot(x,true,color='#1f77b4',lw=.7); ax.plot(x,pred,color='#ff7f0e',ls='--',lw=.7)
                ax.set_title(f'Unit {int(unit)}: {label}',fontsize=8); ax.grid(alpha=.2); ax.set_xlim(0,1)
        fig.suptitle(f'All {dataset} frozen RUL cases', y=.998, fontweight='bold'); fig.tight_layout(rect=(0,0,1,.995))
        fig.savefig(OUT / f'{dataset.lower()}_requested_rul_contact_sheet.png', dpi=220, bbox_inches='tight'); plt.close(fig)
    pd.DataFrame(records).to_csv(OUT/'requested_rul_case_source_data.csv', index=False)
    print({k: len(pd.read_csv(v[0]/'seed_ensemble_predictions_with_intervals.csv').query('model in @v[1]')) for k,v in CONFIG.items()})

if __name__ == '__main__': main()
