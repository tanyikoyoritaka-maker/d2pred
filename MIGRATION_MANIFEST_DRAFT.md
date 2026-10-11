# Proposed migration manifest (no files copied yet)

## Core implementation

- `analysis/expert_exact_d2pred.py`: final Expert architecture and metadata.
- `analysis/theory_aligned_decomposition.py`: causal Wiener/STL trend candidates and trainable trend-fusion network.
- `analysis/theory_aligned_model.py`: positional/token support.
- `analysis/event_segmenters.py`: causal event segmentation and token construction.
- `analysis/causal_decomposition.py`: fold-local causal decomposition primitives.
- `run_ac_nested_gru_rerun.py`: nested device-wise training, checkpoint selection, prediction export, and audits.
- `models/benchmark.py` and model modules used by the approved public baseline registry.

## Protocol and audit code

- `analysis/prepare_ac_cycles_12_16_23_protocol.py` (to be renamed/frozen after audit).
- `analysis/audit_ac_12_16_23_preexecution.py` (to be renamed/frozen after audit).
- `analysis/audit_transformer_repairs.py`.
- `analysis/verify_causal_retraining.py`.
- `analysis/plot_ac_three_scope_rankings.py` and new Figure 3/4 scripts.
- New final scripts for ablation, reliability, robustness, computational cost, supplementary packaging, and checksum verification.

## Data and frozen manifests

- AC raw DI / HI source files and the verified 38-unit inventory.
- `exp_cycles_12_16_review/group_manifest.csv` only after original-record audit passes.
- `exp_cycles_12_16_review/folds.json` only after fold leakage audit passes.
- `environment_audit_draft.json` as the pre-rerun hardware/software record; final rerun must add a lockfile and source revision/checksum.
- IMS, PRONOSTIA, and XJTU-SY source data or documented download manifests.
- Public-data DI caches only if regenerated or checksum-verified from source data.
- Semi-synthetic generator inputs and seeds, not generated outputs alone.

## Explicitly not migrated

- provisional result CSVs and figures from earlier mixed protocols;
- checkpoints selected under incompatible splits;
- LSTM, BiLSTM, and IMDSSN public-baseline results;
- any industrial raw data not authorized for sharing;
- unverified paper tables or manually entered numerical values.

## Functional-failure evidence required before freeze

The source record must identify what “functional failure” means for each industrial
run. The frozen archive must include a non-sensitive citation (record/file/page or
anonymised event table) covering the failure/stop annotation, failure mode or
operating limit, endpoint cycle/time, and whether the trajectory is run-to-failure
or censored. A DI threshold, last available row, or model output is not sufficient
evidence by itself.

## Confirmed industrial cohort

The user confirmed that five, not three, devices are excluded: units `1, 3, 11,
35, 38`. The target frozen cohort is therefore 33 eligible devices, partitioned
as ACCom=16 and ACSim=17.
