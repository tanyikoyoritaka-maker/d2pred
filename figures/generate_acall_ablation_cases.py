"""ACAll preprocessing-ablation case illustration from frozen cache rows."""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures/acall_ablation_cases'
OUT.mkdir(parents=True, exist_ok=True)

def load_unit(unit=20):
    rows = []
    for f in sorted((ROOT / 'results/acall_preprocess_ablation_final/cache').glob('outer_fold_*_rows.pt')):
        rows += torch.load(f, map_location='cpu', weights_only=False)['rows']
    candidates = [r for r in rows if r['split'] == 'test' and int(r['unit']) == unit]
    if not candidates:
        raise RuntimeError(f'unit {unit} is not in the frozen ACAll ablation test cache')
    return sorted(candidates, key=lambda r: (float(r['endpoint']), int(r['prefix_length'])))[-1]

def main():
    unit = 20
    source = load_unit(unit)
    di = np.asarray(source['di'], float)
    full_dwc = np.asarray(source['dwc'], float)
    full_edc = np.asarray(source['edc'], float)
    variants = {
        'Full model': (full_dwc, full_edc),
        'Without EDC': (full_dwc, np.zeros_like(full_edc)),
        'Without decomposition': (di.copy(), np.zeros_like(full_edc)),
    }
    x = np.linspace(0, 1, len(di))
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    records = []
    for j, (name, (dwc, edc)) in enumerate(variants.items()):
        rec = dwc + edc
        for i, (title, vals, color) in enumerate([
            ('Original DI and reconstructed DI', rec, '#d62728'),
            ('DWC', dwc, '#ff7f0e'),
            ('EDC', edc, '#2ca02c'),
        ]):
            ax = axes[i, j]
            if i == 0:
                ax.plot(x, di, color='#1f77b4', lw=1.4, label='Original DI')
                ax.plot(x, vals, color=color, ls='--', lw=1.2, label=r'Reconstructed $\widehat{DI}$')
            else:
                ax.plot(x, vals, color=color, lw=1.3)
            ax.set_title(f'{name}' if i == 0 else title, fontweight='bold')
            ax.grid(alpha=.3); ax.set_xlim(0, 1)
            if i == 2: ax.set_xlabel('Normalized operation cycle')
            if j == 0: ax.set_ylabel('DI value')
        records.append({'unit': unit, 'variant': name,
                        'max_abs_reconstruction_error': float(np.max(np.abs(rec-di))),
                        'reconstruction_rmse': float(np.sqrt(np.mean((rec-di)**2)))})
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .945), ncol=2, frameon=False)
    fig.suptitle(f'ACAll frozen preprocessing-ablation case: anonymized Unit {unit}', y=.995, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(OUT / 'acall_ablation_case_unit20.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'acall_ablation_case_unit20.png', dpi=300, bbox_inches='tight')
    import pandas as pd
    pd.DataFrame(records).to_csv(OUT / 'acall_ablation_case_unit20_metrics.csv', index=False)
    print(records)

if __name__ == '__main__':
    main()
