# D2Pred frozen experiment archive

This file describes the frozen comparison and sensitivity result packages. The
current status of the comparison, sensitivity, and subsequently completed
strict-v2 ablation experiments is recorded in `docs/EXPERIMENT_REGISTRY.md` and
`manifests/active_experiment_registry.json`.

Earlier ablation rejection and approval-pending statements in repository
history describe pre-execution candidates. They remain historical context and
do not supersede the strict-v2 run record, which records 216 completed runs and
the result hash listed in the current registry.

Rejected and superseded experiments remain archived outside this repository
under `../d2pred_rubbish/`. They are not sources for current paper results.
Private industrial DI data, checkpoints, per-prefix predictions, and runtime
caches remain outside Git. The raw-waveform-to-DI mapping is not released.
