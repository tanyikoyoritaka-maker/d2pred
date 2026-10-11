# Recent PRONOSTIA best-result audit

## Located run

The recent experiment is:

`/home/mao/D2Pred/PythonProject2/results/expert_rerun_approved_v1/public_paper_nested`

Its protocol and results were created on 2026-09-16. The model `gru_d2pred_v2` obtained the best RMSE among the old public-baseline comparison on PRONOSTIA:

| model | RMSE | MAE | P90AE | test devices |
|---|---:|---:|---:|---:|
| gru_d2pred_v2 | 23.8166 | 20.4379 | 35.8852 | 16 |
| d2pred_expert_exact_v1 | 24.3638 | 20.7837 | 36.5553 | 16 |

The old baseline comparison was in `public_paper_baselines_nested/summary_metrics.csv`. Thus the statement that “our model ranked first” refers to the historical `gru_d2pred_v2` reimplementation, not the manuscript-exact Expert model.

## Old protocol

- datasets: XJTU-SY and PRONOSTIA;
- models: `gru_d2pred_v2` and `d2pred_expert_exact_v1` for the model run; the companion baseline run contains the old baseline list;
- seeds: 17, 42, 73;
- 3 nested physical-device outer folds;
- 50 epochs, 10 fusion epochs, batch size 8;
- 20 maximum prefixes per unit, sequence length 128;
- prefixes start at 30, stride 20, final endpoint included;
- normalization fit on inner-training devices only;
- checkpoint is the lowest inner-validation primary loss;
- test set is used once after validation selection;
- patience 8 starts after epoch 25; gradient clipping norm 5.

The run has prediction-level outputs, folds, summaries, reliability, computational cost, paired tests and a checksum manifest, so it is substantially more reproducible than the later merged public table.

## Difference from the current candidate

The old run is not directly the current frozen manuscript comparison. It uses a historical `gru_d2pred_v2` architecture, includes LSTM/BiLSTM/IMDSSN in the companion baseline comparison, omits IMS, and uses 10 rather than the proposed 30 fusion epochs. The current merged public values also combine different source runs and do not demonstrate one common manifest.

## Decision

The old result is **usable as an audited historical result**, but **not usable as the current frozen manuscript benchmark** without an explicit model-list decision and protocol harmonisation. A current comparison rerun remains necessary and is **not started**. The machine-readable record is `pronostia_best_old_experiment_audit.json`.
