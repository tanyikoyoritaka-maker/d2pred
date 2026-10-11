from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / 'eaaipaper' / 'fig' / 'Figure_3.pdf'
DATA = ROOT / 'figures' / 'ac_case_studies_frozen' / 'ac_di_decomposition_source_data.csv'
SELECTION = {'ACSim': 10, 'ACCom': 12, 'ACAll': 20}
SOURCES = {'ACSim': ROOT / 'results' / 'final_ac_nested',
           'ACCom': ROOT / 'results' / 'final_ac_nested',
           'ACAll': ROOT / 'results' / 'acall_nested'}

def load_row(source, group, unit):
    candidates = []
    for f in sorted((source / 'cache').glob('outer_fold_*_rows.pt')):
        rows = torch.load(f, map_location='cpu', weights_only=False)['rows']
        for row in rows:
            if row.get('split') == 'test' and int(row['unit']) == unit and (group == 'ACAll' or row.get('group') == group):
                candidates.append(row)
    if not candidates:
        raise RuntimeError(f'No frozen test row for {group} unit {unit}')
    return max(candidates, key=lambda r: (float(r['endpoint']), int(r['prefix_length'])))

def main():
    rows = {g: load_row(SOURCES[g], g, u) for g, u in SELECTION.items()}
    fig, axes = plt.subplots(3, 3, figsize=(13, 7.6), sharex='col')
    titles = {'ACSim': 'ACSim', 'ACCom': 'ACCom', 'ACAll': 'ACAll'}
    for i, group in enumerate(('ACSim', 'ACCom', 'ACAll')):
        row = rows[group]
        di = np.asarray(row['di'], float); dwc = np.asarray(row['dwc'], float); edc = np.asarray(row['edc'], float)
        x = np.linspace(0, 1, len(di))
        axes[i, 0].plot(x, edc, color='#2ca02c', lw=1.3)
        axes[i, 1].plot(x, dwc, color='#ff7f0e', lw=1.3)
        axes[i, 2].plot(x, di, color='#1f77b4', lw=1.3)
        axes[i, 0].set_ylabel('Value')
        for j in range(3):
            axes[i, j].grid(alpha=.3, lw=.5); axes[i, j].set_xlim(0, 1)
        axes[i, 0].set_title(f'({chr(97+i*3)}) {titles[group]} Unit {SELECTION[group]}: EDC', fontweight='bold', fontsize=10)
        axes[i, 1].set_title(f'({chr(98+i*3)}) {titles[group]} Unit {SELECTION[group]}: DWC', fontweight='bold', fontsize=10)
        axes[i, 2].set_title(f'({chr(99+i*3)}) {titles[group]} Unit {SELECTION[group]}: DI', fontweight='bold', fontsize=10)
    for ax in axes[-1, :]: ax.set_xlabel('Operation cycle')
    fig.suptitle('Frozen AC decomposition examples', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .96), w_pad=1.1, h_pad=1.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches='tight')
    plt.close(fig)
    # Keep a traceable source table for the manuscript figure.
    import csv
    with DATA.open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['group', 'unit', 'component', 'x', 'value'])
        for group in ('ACSim', 'ACCom', 'ACAll'):
            row = rows[group]; arrays = [('edc', row['edc']), ('dwc', row['dwc']), ('di', row['di'])]
            for component, values in arrays:
                for x, value in zip(np.linspace(0, 1, len(values)), values):
                    writer.writerow([group, SELECTION[group], component, float(x), float(value)])
    print(SELECTION)

if __name__ == '__main__': main()
