# D2Pred final complete experiment freeze protocol v1

Status: **frozen and approved for execution on 2026-09-22**.

This protocol adds an independent pooled ACAll experiment and pre-registers all comparison, ablation and sensitivity work. It contains no provisional numerical results.

## 1. Common protocol

All experiments use seeds 17, 42 and 73, three nested physical-device outer folds, device-wise inner validation, cycle-based RUL targets, training-only normalization, sequence length 128, prefix start 30, prefix stride 20, final endpoint included, at most 20 prefixes/unit, Adam with learning rate 0.001, batch size 8, 50 epochs, 50 fusion epochs, validation-loss checkpoint selection, patience 8 after epoch 25 and gradient clipping norm 5.0. Test data are evaluated once and never used for tuning.

Primary metrics are RMSE, MAE and P90AE in cycles. Every run must emit prediction-level files, per-device metrics, fold/seed summaries, paired tests, reliability, computational cost, environment information and checksums.

## 2. Dataset partition and experiments

The eligible industrial devices are partitioned by one fixed, pre-registered rule. For every device (u), let (L_u) be its endpoint lifetime in cycles:

\[
G(u)=\begin{cases}
\mathrm{ACSim}, & L_u<1000,\\
\mathrm{ACCom}, & L_u\geq1000.
\end{cases}
\]

This rule is exhaustive and has no manual exception or post-hoc reassignment. The verified effective assignment contains 17 ACSim devices and 16 ACCom devices. A new independent ACAll condition pools all 33 eligible AC units. ACAll does not distinguish ACSim from ACCom during training or evaluation; those labels remain metadata only. ACAll must have its own device-wise folds generated and frozen before training.

The same comparison is run on IMS, PRONOSTIA and XJTU-SY. The model list is D2Pred manuscript-exact Expert plus CNN, DAG_network, PM2FN, Transformer, TCN_BiGRU, KGHM, ATCN, MDFA, SBi_Transformer and Attn_ResDSCNN.

## 3. Ablation study

The manuscript-exact Expert is the reference. The registered ablation mechanisms are: without Wiener, without rolling-prefix STL, without learned autoencoder trend fusion, without DWC, without EDC (including its AEP tokens), without DWC/EDC decomposition, and without AEP. The full model is included as the reference row. Each condition changes one registered preprocessing element while retaining the same Expert predictor wherever the input remains defined. Ablations run separately on ACSim, ACCom and ACAll under identical folds, seeds, target, tuning budget and stopping rules.

## 4. Sensitivity study

After the preprocessing ablation has completed and passed its audit, one-factor-at-a-time sensitivity is run separately on ACSim, ACCom and ACAll. The seven frozen factors are: segmentation sensitivity \(\gamma\) (1.5/2.0/2.5/3.0/3.5), minimum segment length (4/8/12/16), maximum accepted change points (8/15/24), maximum output tokens (8/16/24), prefix start cycle (10/30/50), prefix stride (10/20/40), and maximum prefixes per unit (10/20/40). All other settings, folds, seeds, target definition, normalization, checkpoint rule, stopping rule and metrics remain unchanged. Every registered value is reported; no test result may be used to select a preferred value.

## 5. Final visualisation

Only after all runs pass completeness and NaN/leakage audits, one multi-page PDF and matching PNG dashboard will be generated directly from verified prediction files. It will include ACAll/ACSim/ACCom and public rankings, numerical tables, ablation deltas, sensitivity curves, uncertainty/reliability and computational cost. Hand-entered values are prohibited.

Machine-readable version: `complete_experiment_freeze_protocol_v1.json`.
