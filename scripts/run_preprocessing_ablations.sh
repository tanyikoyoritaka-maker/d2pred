#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/mao/D2Pred/d2pred_final
PYTHONPATH=/home/mao/D2Pred/d2pred_final:/home/mao/D2Pred/PythonProject2
export PYTHONPATH
PYTHON=/home/mao/D2Pred/PythonProject2/.venv-linux/bin/python
RUNNER="$ROOT/run_ac_nested_gru_rerun.py"
MODELS=d2pred_expert_exact_v1,ablation_no_wiener,ablation_no_stl,ablation_no_autoencoder_fusion,ablation_preprocess_no_dwc,ablation_preprocess_no_edc,ablation_preprocess_no_decomposition,ablation_no_aep
COMMON=(--profile core --models "$MODELS" --outer-folds 3 --seeds 17,42,73 --epochs 50 --fusion-epochs 50 --batch-size 8 --device cuda --shared-decomposition --group-manifest "$ROOT/manifests/group_manifest.csv")

run_one() {
  local output=$1
  shift
  if [[ -f "$output/run_status.json" ]] && grep -q 'verified_complete' "$output/run_status.json"; then
    echo "[skip verified complete] $output"
    return
  fi
  "$PYTHON" -u "$RUNNER" --output "$output" "${COMMON[@]}" "$@"
}

run_one "$ROOT/results/acsim_preprocess_ablation_final" --only-group ACSim
run_one "$ROOT/results/accom_preprocess_ablation_final" --only-group ACCom
run_one "$ROOT/results/acall_preprocess_ablation_final" --mix-groups --folds-file "$ROOT/manifests/folds.json"

