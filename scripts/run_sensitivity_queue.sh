#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/mao/D2Pred/d2pred_final
PYTHON=/home/mao/D2Pred/PythonProject2/.venv-linux/bin/python
export PYTHONPATH=$ROOT:/home/mao/D2Pred/PythonProject2
RUNNER=$ROOT/run_ac_nested_gru_rerun.py
run_case() {
  local ds=$1 factor=$2 value=$3; shift 3
  local out=$ROOT/results/sensitivity_${ds}_${factor}_${value}
  if [[ -f $out/run_status.json ]] && grep -q verified_complete $out/run_status.json; then return; fi
  "$PYTHON" -u "$RUNNER" --output "$out" --profile core --models d2pred_expert_exact_v1 --outer-folds 3 --seeds 17,42,73 --epochs 50 --fusion-epochs 50 --batch-size 8 --device cuda --shared-decomposition --group-manifest $ROOT/manifests/group_manifest.csv "$@"
}
run_dataset() {
  local ds=$1 group=$2 extra=()
  [[ $ds == acall ]] && extra=(--mix-groups --folds-file $ROOT/manifests/folds.json)
  [[ $ds != acall ]] && extra=(--only-group "$group")
  for v in 1.5 2.0 2.5 3.0 3.5; do run_case "$ds" gamma "$v" "${extra[@]}" --aep-gamma "$v"; done
  for v in 4 8 12 16; do run_case "$ds" min_size "$v" "${extra[@]}" --aep-min-size "$v"; done
  for v in 8 15 24; do run_case "$ds" max_points "$v" "${extra[@]}" --aep-max-points "$v"; done
  for v in 8 16 24; do run_case "$ds" max_tokens "$v" "${extra[@]}" --aep-max-tokens "$v"; done
  for v in 10 30 50; do run_case "$ds" prefix_start "$v" "${extra[@]}" --prefix-start-cycle "$v"; done
  for v in 10 20 40; do run_case "$ds" prefix_stride "$v" "${extra[@]}" --prefix-stride "$v"; done
  for v in 10 20 40; do run_case "$ds" max_prefixes "$v" "${extra[@]}" --maximum-prefixes "$v"; done
}
run_dataset acsim ACSim
run_dataset accom ACCom
run_dataset acall ACAll
