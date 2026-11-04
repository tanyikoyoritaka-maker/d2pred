# D2Pred

The main comparison, sensitivity, and strict-v2 ablation experiments have
completed on the server. Their completion evidence and result locations are
recorded in [the experiment registry](docs/EXPERIMENT_REGISTRY.md) and
`manifests/active_experiment_registry.json`.

Start with:

- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Repository structure](docs/REPOSITORY_STRUCTURE.md)
- [Experiment registry](docs/EXPERIMENT_REGISTRY.md)

The canonical compact ablation result is `ablation_final/metrics_summary.csv`.
Its SHA-256 is `6a1d967ca1d1b8d04428404da06a6572edb5c02459586b3c59f1d2ad0fde6588`.

Private industrial data, checkpoints, full prediction files, and runtime caches
are retained on the server and excluded from Git. Restore the private DI dataset
described by `manifests/data_source_manifest.json` before attempting a reproduction.
