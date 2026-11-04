# Reproducibility

This guide records available entrypoints and frozen settings. Commands are for
a clean clone or a separate output archive. Do not point them at the server's
frozen result directories: training and evaluation scripts write to their output
locations.

## Environment

Use Python 3.11 and the pins in `requirements-lock.txt`. The recorded run used
PyTorch 2.6.0+cu124 with CUDA runtime 12.4 on an NVIDIA RTX 2080 Ti. The
`Containerfile` defines the CUDA 12.4.1 runtime image. Industrial reproduction
requires access to the private DI dataset, which is not included in Git.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements-lock.txt
```

Restore `data/industrial/DI_df.csv` from the approved private archive and verify
its hash against `manifests/data_source_manifest.json` and the experiment registry.
Do not publish that file or the source waveform-to-DI mapping.

## Comparison

The frozen settings use `manifests/group_manifest.csv` and `manifests/folds.json`,
seeds `17,42,73`, three outer folds, up to 50 epochs, 50 fusion epochs, and batch
size 8. In a separate reproduction archive, run the main runner once per dataset
scope with a new output path:

```sh
python -u run_ac_nested_gru_rerun.py --output results/reproduction/comparison/ACSim --profile full --models d2pred_expert_exact_v1,CNN,DAG_network,PM2FN,Transformer,TCN_BiGRU,KGHM,ATCN,MDFA,SBi_Transformer,Attn_ResDSCNN --outer-folds 3 --seeds 17,42,73 --epochs 50 --fusion-epochs 50 --batch-size 8 --device cuda --group-manifest manifests/group_manifest.csv --folds-file manifests/folds.json --only-group ACSim
python -u run_ac_nested_gru_rerun.py --output results/reproduction/comparison/ACCom --profile full --models d2pred_expert_exact_v1,CNN,DAG_network,PM2FN,Transformer,TCN_BiGRU,KGHM,ATCN,MDFA,SBi_Transformer,Attn_ResDSCNN --outer-folds 3 --seeds 17,42,73 --epochs 50 --fusion-epochs 50 --batch-size 8 --device cuda --group-manifest manifests/group_manifest.csv --folds-file manifests/folds.json --only-group ACCom
python -u run_ac_nested_gru_rerun.py --output results/reproduction/comparison/ACAll --profile full --models d2pred_expert_exact_v1,CNN,DAG_network,PM2FN,Transformer,TCN_BiGRU,KGHM,ATCN,MDFA,SBi_Transformer,Attn_ResDSCNN --outer-folds 3 --seeds 17,42,73 --epochs 50 --fusion-epochs 50 --batch-size 8 --device cuda --group-manifest manifests/group_manifest.csv --folds-file manifests/folds.json --mix-groups
```

This is a multi-model CUDA workload. The archived execution used an RTX 2080 Ti;
allow several hours for all three scopes. Compare each output's `run_status.json`,
`summary_metrics.csv`, `folds.json`, and `checksum_manifest.csv` with the
registered package.

## Sensitivity

The one-factor runner consumes comparison predictions from its sibling
`comparison/` path. Run it only in a separate reproduction tree after producing
the comparison package:

```sh
python scripts/run_sensitivity_independent_final.py --root-output results/reproduction/sensitivity
```

It uses the values and defaults in the result `execution_manifest.json` and
experiment registry. It launches 57 non-default model runs, so it requires
substantial CUDA time. Verify the three dataset CSVs and
`sensitivity_summary_all_metrics.csv` against the registered parameter grid and
per-case status/checksum files.

To regenerate a sensitivity curve from the registered aggregate:

```sh
python figures/plot_registered_sensitivity.py --output figures/registered_sensitivity.pdf
```

To save the figure into the sibling paper project, pass an explicit output such
as `--output ../eaaipaper/fig/registered_sensitivity.pdf`.

## Ablation

First create the comparison package in the same clean reproduction tree. Restore
the private DI data, then run the registered strict-v2 batch and evaluator:

```sh
python ablation_final/run_strict_ablation_batch.py
python ablation_final/evaluate_strict_ablation.py
```

The batch runs eight variants over three datasets, three outer folds, and seeds
17, 42, and 73 (216 trainings). Full D2Pred predictions are reused from the
comparison package. The archived run took about four hours on the recorded RTX
2080 Ti, in addition to the comparison run. These scripts use
`ablation_final/outputs/strict_ablation_v2/`; use a separate clone or output
archive so server results remain untouched.

Check the evaluator manifest's `metrics_output_sha256` and the canonical result:

```sh
sha256sum ablation_final/metrics_summary.csv
```

Expected SHA-256:
`52fff4dfeae742cbfd79140f7a0cfa5e3fa0d47bb01745c15f7f5fbddc2f56a1`.

The evaluator recomputes metrics and writes outputs. Do not run it against the
frozen server result directory just to validate the archived CSV.

## Source Revision Limits

Strict-v2 ablation records bind a base commit, a working-tree diff hash, and
per-file fingerprints. Comparison and sensitivity execution records do not
contain execution-time source fingerprints. The repository records this gap
instead of presenting the current Git commit as proof of the historical source.

## Result Checks

For the frozen server package, compare the official ablation CSV's SHA-256 with
the value above. Comparison outputs have per-dataset run statuses and checksum
manifests; sensitivity has an execution manifest, aggregate checksum manifest,
and per-case statuses. Use these existing records; do not rerun training to
validate them.
