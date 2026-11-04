# Repository Structure

| Path | Classification | Contents |
|---|---|---|
| `analysis/`, `models/`, `public_processing/`, `configs/`, `scripts/` | Active | Model, preprocessing, benchmark, evaluation, and audit sources. |
| `ablation_final/` | Active | Strict-v2 implementation, protocol, official compact result, and server-side run outputs. |
| `results/frozen_protocol_final/comparison/` | Active | Main comparison summaries and server-side predictions, checkpoints, logs, and checksums. |
| `results/frozen_protocol_final/sensitivity/` | Active | Registered sensitivity aggregates and server-side case outputs. |
| `results/public_benchmark_final/`, `results/peak_memory_supplement/` | Supplementary | Separate public benchmark and computational evidence packages. |
| `../d2pred_rubbish/` | Quarantined / historical | Rejected and superseded experiment outputs and source snapshots, retained outside this repository. |
| `__pycache__/`, ignored caches and runtime logs | Temporary | Generated local artifacts; not tracked. |
| `healthy_sensitivity/`, `supplementary_sensitivity/`, `new_parameter/`, `supplement/` | Unknown / supplementary | Retained until their relationship to current paper results is established. |
| `../eaaipaper/` | External sibling project | Paper source, generated assets, and PDF. Outside this Git repository; unchanged by this update. |

Git tracks compact comparison summaries/statuses/checksums, the sensitivity
aggregate and execution manifest, and ablation metrics/evaluation manifests.
Full predictions, checkpoints, caches, private data, and most runtime logs remain
on the server. No result or history directory was moved or deleted.

The root `checksum_manifest.csv` and 19 pre-existing deleted paths remain
uncommitted worktree changes. They were present before this archive update and
are excluded from its commit.
