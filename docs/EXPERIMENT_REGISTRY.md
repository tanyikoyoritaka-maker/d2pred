# D2Pred Experiment Registry

**Current server status:** Comparison Experiments, Sensitivity Experiments, and
Ablation Experiments are `COMPLETED_AND_VERIFIED`. This status reflects the
current result files and their execution records. Earlier approval-pending
entries in historical protocol snapshots describe pre-execution candidates.

The machine-readable registry is `manifests/active_experiment_registry.json`.
The primary result packages live under `results/frozen_protocol_final/`; large
run artifacts are retained on the server and omitted from Git.

| Experiment | Status | Result source | Existing completion evidence |
|---|---|---|---|
| Comparison Experiments | `COMPLETED_AND_VERIFIED` | `results/frozen_protocol_final/comparison/{ACSim,ACCom,ACAll}/summary_metrics.csv` | Per-dataset `run_status.json` says `verified_complete`; output checksum manifests and frozen folds are present. |
| Sensitivity Experiments | `COMPLETED_AND_VERIFIED` | `results/frozen_protocol_final/sensitivity/sensitivity_summary_all_metrics.csv` and dataset summaries | `execution_manifest.json` says `verified_complete`, records 57 non-default runs and 78 rows; checksums and per-case status records are present. |
| Ablation Experiments | `COMPLETED_AND_VERIFIED` | `ablation_final/metrics_summary.csv` | SHA-256 matches the evaluation manifest; the batch record contains 216/216 `COMPLETED` runs with exit code 0 and artifact hashes; all ten code fingerprints match committed source. |

## Comparison Experiments

- Entrypoint: `run_ac_nested_gru_rerun.py`.
- Results: `results/frozen_protocol_final/comparison/{ACSim,ACCom,ACAll}/`.
- Per-dataset outputs: `summary_metrics.csv`, frozen predictions, validation
  predictions, `run_status.json`, `folds.json`, `checksum_manifest.csv`, and
  run/fusion checkpoints.
- Dataset groups: ACSim (17 devices), ACCom (16), and ACAll (33).
- Seeds: 17, 42, 73; outer folds: 3; maximum epochs: 50; batch size: 8.
- Model list and remaining settings are in each result `protocol.json`.
- Full D2Pred values used by the ablation comparison come only from these main
  comparison `test_predictions.csv` files. The ablation Full rows reuse those
  predictions and are not a second Full training run.
- Verification: all three per-dataset `run_status.json` files say
  `verified_complete`; recorded test-prediction counts are 11,154, 10,560, and
  21,714 respectively.
- Source revision limitation: the execution records contain commands and
  output checksums, but no execution-time Git commit or source-code fingerprint.
  The present source is in Git; its identity as the exact historical execution
  source cannot be established from these records alone.

## Sensitivity Experiments

- Entrypoint: `scripts/run_sensitivity_independent_final.py`; base runner:
  `run_ac_nested_gru_rerun.py`.
- Results: `results/frozen_protocol_final/sensitivity/`.
- Summary outputs: `sensitivity_summary_all_metrics.csv`,
  `sensitivity_ACSim.csv`, `sensitivity_ACCom.csv`, and `sensitivity_ACAll.csv`.
- Factors and values from the execution manifest: gamma `[1.5, 2.0, 2.5, 3.0,
  3.5]`; minimum segment length `[4, 8, 12, 16]`; maximum change points `[8,
  15, 24]`; maximum tokens `[8, 12, 16, 24, 32]`; prefix start `[10, 30, 50]`;
  prefix stride `[10, 20, 40]`; maximum prefixes `[10, 20, 40]`. Defaults are
  2.5, 8, 15, 16, 30, 20, and 20 respectively.
- Seeds: 17, 42, 73; outer folds: 3. The default points reuse main-comparison
  predictions; the execution manifest records 57 non-default runs and 78
  summary rows over the three datasets.
- Checkpoints and predictions are under each case directory's `runs/` tree;
  case protocols, commands, run statuses, and checksums remain in the server
  result archive.
- Source revision limitation: sensitivity result manifests do not bind a
  source commit or code fingerprint. The registered current runner is identified
  above, but the exact historical source revision is not independently proven.
- The plotting entrypoint reads the registered dataset summaries. The generated
  paper figure is in sibling `../eaaipaper/fig/`, outside this Git repository.

## Ablation Experiments

- Official result: `ablation_final/metrics_summary.csv` (4,972 bytes).
- SHA-256: `6a1d967ca1d1b8d04428404da06a6572edb5c02459586b3c59f1d2ad0fde6588`.
- The evaluation output at
  `ablation_final/outputs/strict_ablation_v2/evaluation/metrics_summary.csv`
  has the same byte hash.
- Datasets: ACSim, ACCom, ACAll. Each has one frozen Full reference plus eight
  variants: `no_wiener`, `no_stl`, `no_autoencoder_fusion`, `no_dwc`, `no_edc`,
  `no_decomposition`, `no_fusion`, and `no_aep`.
- Metrics: `rmse_cycles`, `mae_cycles`, `p90ae_cycles` (27 data rows; nine per
  dataset). The CSV uses three outer folds, seeds 17/42/73, averages seeds per
  device, then takes an equal-device mean.
- Entrypoints: `ablation_final/run_strict_ablation_batch.py` and
  `ablation_final/evaluate_strict_ablation.py`.
- Outputs: `ablation_final/outputs/strict_ablation_v2/`. Checkpoints, training
  logs, validation/test predictions, and preprocessing caches remain on the
  server and are ignored by Git. The batch and evaluation manifests record their
  paths and hashes.
- Full reference: frozen predictions from the main comparison package; no Full
  model was retrained for the ablation.
- The checked-in v2 protocol has repository-relative file locations. The exact
  executed protocol SHA-256 is retained in the machine-readable registry; only
  absolute server path prefixes were normalized in the public copy.
- Run `strict_v2_20261103T090504Z_1de96183`: 216 expected and completed runs,
  three folds, three seeds, eight variants, and three datasets. Its ten recorded
  code fingerprints match the committed source. The execution used base commit
  `51c696597efc5e436ad14a73b8fcd026907bd542` plus the working-tree diff hash
  recorded in the machine-readable registry.
- Full-mode metrics are the verified reference values in the official CSV.
  The evaluation manifest records the same SHA-256 as the canonical file.

## Data and Environment

The private source is `data/industrial/DI_df.csv`; its recorded SHA-256 is
`cc32bbd6e0f3cb7f2e30422dacb821fb13eb2b2bb0b13956cf138a1651d1dd82`.
It is ignored by Git. The tracked split definitions are
`manifests/group_manifest.csv`, `manifests/failure_endpoint_manifest.csv`, and
`manifests/folds.json`. Comparison and ablation records contain corresponding
fold/data hashes.

The recorded environment is Python 3.11.15, PyTorch 2.6.0+cu124, NumPy 1.26.4,
pandas 2.2.3, CUDA runtime 12.4, and an NVIDIA RTX 2080 Ti. Package pins are in
`requirements-lock.txt`; the container base is described in `Containerfile`.

## Historical and Quarantined Material

Earlier approval-pending entries remain in Git history and older protocol
documents. They describe pre-execution candidates and are superseded for current
completion reporting by the strict-v2 execution and evaluation records.
Rejected v3 outputs and older candidate runs remain outside this repository in
`../d2pred_rubbish/`; they are not current result sources. This update did not
delete or promote historical results.

## Manuscript and Public-Path Issues

The sibling paper project is outside this repository. Its active ablation
section now reports the strict-v2 results and defines w/o AEP as removing
adaptive event partitioning while retaining the event-token pathway: the causal
EDC prefix is represented by one statistics token and processed by the Event
Transformer with cross-branch attention. The result table is aligned with the
official CSV at the displayed precision.

The committed paper checksum manifest is stale for the sibling paper build:
the recorded `main.pdf` size/hash does not match the current sibling PDF, and
the listed `ACAll_ablation_extreme_cases.pdf` is absent there. Neither mismatch
changes the experiment result status.
