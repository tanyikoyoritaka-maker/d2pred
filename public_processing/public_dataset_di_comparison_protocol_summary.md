# Public dataset DI comparison protocol

- Dataset paths: see `dataset_path_manifest.json`.
- Bearing DI: vibration features, per-unit Spearman averaged absolute correlation, top-14, training StandardScaler, PCA(2), z1.
- Battery DI: `1-capacity/C0`, C0 from first five valid discharge cycles, strict 70% threshold endpoint.
- IMS: only four failures documented in the bundled PDF are used.
- Splits are unit-level 60/20/20 where feasible; IMS uses a tiny-unit split.
- DWC = autoencoder fusion of Wiener and STL trends; EDC = DI-DWC.
- `cpm_stats_tokens_dwc_cross_attn` is excluded. No FD, AC, or ablation experiment is run.
