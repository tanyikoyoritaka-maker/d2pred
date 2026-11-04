import argparse
from pathlib import Path
import csv
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SENSITIVITY_ROOT = ROOT / 'results/frozen_protocol_final/sensitivity'
DATASETS = [('ACSim', 'acsim'), ('ACCom', 'accom'), ('ACAll', 'acall')]
FACTORS = [('gamma', r'$\gamma$', 2.5), ('min_size', 'Minimum segment length $m$', 8),
           ('max_points', r'Maximum change points $K_{cp}$', 15),
           ('max_tokens', r'Maximum output tokens $K$', 16), ('prefix_start', 'Prefix start', 30),
           ('prefix_stride', 'Prefix stride', 20), ('max_prefixes', 'Maximum prefixes per device', 20)]

def read_factor(dataset, factor):
    path = SENSITIVITY_ROOT / ('sensitivity_%s.csv' % dataset)
    with path.open(newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if row['factor'] == factor]
    return sorted((float(row['value']), float(row['device_mean_RMSE_cycles']),
                   float(row['RMSE_95CI_low']), float(row['RMSE_95CI_high'])) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'figures/registered_sensitivity.pdf')
    args = parser.parse_args()

    fig, axes = plt.subplots(7, 1, figsize=(8.2, 15.0), constrained_layout=True)
    colors = {'ACSim': '#1f77b4', 'ACCom': '#d62728', 'ACAll': '#2ca02c'}
    for ax, (factor, xlabel, selected) in zip(axes, FACTORS):
        for label, dataset in DATASETS:
            values = read_factor(dataset, factor)
            x = [v[0] for v in values]; y = [v[1] for v in values]
            lo = [v[2] for v in values]; hi = [v[3] for v in values]
            ax.plot(x, y, marker='o', linewidth=1.5, markersize=3.5, label=label, color=colors[label])
            ax.fill_between(x, lo, hi, color=colors[label], alpha=0.12)
        ax.axvline(selected, color='black', linestyle='--', linewidth=0.9)
        ax.set_ylabel('RMSE'); ax.set_xlabel(xlabel); ax.grid(alpha=0.25)
    axes[0].legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.35), frameon=False)
    fig.suptitle('One-factor sensitivity of D$^2$Pred to the registered AEP and prefix-generation parameters', fontsize=12, y=1.01)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches='tight')


if __name__ == '__main__':
    main()
