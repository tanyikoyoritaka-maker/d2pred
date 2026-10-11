# Public benchmark provenance audit

## Decision

The currently visualised public benchmark values are **not frozen**. They are traceable to existing files, but they are assembled from more than one completed run and therefore do not satisfy a single frozen common-protocol record.

## What the current values use

- Retained-baseline rows: `PythonProject2/exp_public_baselines_repaired_transformer/merged_results/all_public_baselines_metrics.csv`.
- Expert rows: `PythonProject2/results/new_standard_rerun_20260915/runs/{ims,pronostia,xjtu_sy}/expert_aligned_aep_gated_causal/metrics.json`.
- Ranking/merge code: `PythonProject2/analysis/build_overall_expert_public_ranking.py`.
- Repaired Transformer evidence: `exp_public_baselines_repaired_transformer/transformer_rerun_v3`, which is a three-seed Transformer-only rerun.

The retained visualisation contains 3 datasets × 11 models (10 retained baselines plus Expert). LSTM, BiLSTM and IMDSSN were excluded from the visualisation, as requested.

## Protocol evidence found

The available protocol files specify many useful settings: seed 42 for the earlier all-model public-baseline run, 50 training epochs, 30 fusion epochs, batch size 8, maximum 20 prefixes per device, sequence length 128, training-only normalization, and validation-MSE checkpoint selection. The Transformer repair protocol specifies seeds 17/42/73 and three outer folds.

## Why it cannot be frozen

1. The all-model baseline values are single-seed, while the repaired Transformer run is three-seed and only covers Transformer.
2. Expert values come from a different new-standard run than the repaired Transformer evidence.
3. The human-review protocol records a model list and status but not the complete public folds, target/endpoint manifests, environment checksum, or per-run provenance.
4. The upstream priority-2 protocol still lists LSTM, BiLSTM and IMDSSN, conflicting with the later removal decision.
5. No single checksum archive covers the public code, data processing, folds, manifests and merged table.

## Proposed rerun for approval

Run IMS, PRONOSTIA and XJTU-SY for D2Pred Expert plus the ten retained baselines, using identical device-wise nested folds, seeds 17/42/73, 50 epochs, 30 fusion epochs, batch size 8, training-only normalization, and validation-only checkpoint selection. Freeze public endpoint/target manifests and folds before execution, and emit prediction-level files, metrics, environment, manifests and SHA-256 checksums.

This proposed run has **not** been started. See the machine-readable audit and proposal in `public_benchmark_protocol_audit.json`.
