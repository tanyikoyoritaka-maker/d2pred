# D2Pred final public benchmark freeze protocol v1

Status: **proposed, not executed, awaiting human approval**.

This protocol replaces all provisional public benchmark result files previously produced in `d2pred_final`. It contains no benchmark numbers.

## Scope

Datasets: IMS, PRONOSTIA and XJTU-SY.

Models: D2Pred manuscript-exact Expert plus CNN, DAG_network, PM2FN, Transformer, TCN_BiGRU, KGHM, ATCN, MDFA, SBi_Transformer and Attn_ResDSCNN.

## Frozen evaluation

- three nested physical-device outer folds;
- identical folds for every model and seed;
- seeds 17, 42 and 73;
- folds and endpoint/target manifests frozen before any model fitting;
- unit split occurs before prefix generation;
- RUL target is `endpoint_cycle - current_cycle` in cycles;
- prefix start cycle 30, stride 20, final endpoint included, maximum 20 prefixes/unit;
- sequence length 128 and identical observed prefix endpoints across models;
- normalization uses inner-training units only;
- test units are used exactly once after validation-based checkpoint selection.

## Training

Adam, learning rate 0.001, batch size 8, maximum 50 epochs, 30 fusion epochs, learning rate 1e-3 through epoch 25 followed by 5e-4, patience 8 beginning after epoch 25, gradient clipping norm 5.0. Checkpoints are selected only by lowest inner-validation primary loss. No test-based tuning or model-specific extra search is allowed.

The Expert implementation must satisfy the manuscript specification: trend-fusion network, gradual-component encoder, twelve raw residual descriptors, token projection, three masked self-attention blocks, attention-guided fusion, Softplus output and joint loss. Adapted baselines must be labelled in metadata and use the same folds, seeds, target, tuning budget and stopping rules.

## Freeze gate

Before execution, create and hash the dataset manifest, endpoint manifest, folds, model registry, environment, configuration and source code. Do not start training until a human approves this exact protocol. After approval, generate prediction-level outputs, per-device metrics, fold/seed summaries, paired tests, reliability, computational cost and a final checksum manifest. Figures may only be generated from those verified prediction files.

Machine-readable version: `public_benchmark_freeze_protocol_v1.json`.
