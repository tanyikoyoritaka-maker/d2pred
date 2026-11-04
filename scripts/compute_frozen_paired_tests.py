"""Compute registered device-level paired Wilcoxon tests for frozen outputs."""

from pathlib import Path
import argparse

import pandas as pd
from scipy.stats import wilcoxon


REFERENCE = "d2pred_expert_exact_v1"
METRICS = {
    "RMSE_cycles": "RMSE",
    "MAE_cycles": "MAE",
    "P90AE_cycles": "P90AE",
}


def holm(values):
    """Return Holm-adjusted p-values in the original order."""
    p = pd.Series(values, dtype=float)
    order = p.sort_values(kind="mergesort").index.tolist()
    adjusted = pd.Series(index=p.index, dtype=float)
    running = 0.0
    n = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, float(p.loc[idx]) * (n - rank))
        running = max(running, value)
        adjusted.loc[idx] = running
    return adjusted


def compute(path):
    unit = pd.read_csv(path / "device_level_metrics.csv")
    # A device is the statistical unit.  Seeds are averaged within each
    # device; prefixes never enter the paired test as independent rows.
    unit = (
        unit.groupby(["model", "group", "unit"], as_index=False)[
            list(METRICS)
        ].mean()
    )
    records = []
    for group in sorted(unit["group"].unique()):
        group_values = unit[unit["group"] == group]
        pivot = group_values.pivot(index="unit", columns="model", values=list(METRICS))
        if REFERENCE not in pivot.columns.get_level_values(1):
            continue
        for column, label in METRICS.items():
            if pivot[column].isna().any().any():
                raise ValueError(f'Incomplete device pairing: {path}, {group}, {label}')
            ref = pivot[(column, REFERENCE)]
            comparisons = []
            for model in sorted(set(pivot[column].columns) - {REFERENCE}):
                pair = pd.concat([ref.rename("reference"), pivot[(column, model)].rename("comparison")], axis=1).dropna()
                difference = pair["comparison"] - pair["reference"]
                if len(difference) < 2 or (difference != 0).sum() == 0:
                    statistic, p_value = 0.0, 1.0
                else:
                    statistic, p_value = wilcoxon(
                        difference.to_numpy(), alternative="two-sided",
                        zero_method="wilcox", method="auto",
                    )
                comparisons.append({
                    "group": group,
                    "metric": label,
                    "reference": REFERENCE,
                    "comparison": model,
                    "num_paired_devices": int(len(pair)),
                    "mean_difference_comparison_minus_reference": float(difference.mean()),
                    "wilcoxon_statistic": float(statistic),
                    "p_value_two_sided": float(p_value),
                })
            if comparisons:
                adjusted = holm([row["p_value_two_sided"] for row in comparisons])
                for idx, row in enumerate(comparisons):
                    row["holm_p_value"] = float(adjusted.iloc[idx])
                    row["holm_significant_0.05"] = bool(adjusted.iloc[idx] <= 0.05)
                    records.append(row)
    result = pd.DataFrame(records)
    result.to_csv(path / "paired_statistical_tests.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    for directory in args.directories:
        result = compute(directory)
        print(directory, len(result), "paired tests")


if __name__ == "__main__":
    main()
