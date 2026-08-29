#!/usr/bin/env bash
set -euo pipefail

study_root=${RIG_RECON_BETA_STUDY_ROOT:-/tmp/GPT-speedrun-TPU-fuzzy-reconstruction-auxk}
artifact_root=${RIG_RECON_BETA_RUNS_ROOT:-${study_root}/runs}
study_id=fuzzy-topk-reconstruction-beta-sweep-125m-v4-3seed
suite_path=recipes/fuzzy_topk_reconstruction/study-suite-125m-v4-reconstruction-beta-sweep-3seed.json

cd "${study_root}"
test -z "$(git status --porcelain --untracked-files=no)"
suite_sha=$(sha256sum "${suite_path}" | cut -d' ' -f1)
mkdir -p "${artifact_root}"
exec > >(tee -a "${artifact_root}/beta-sweep-queue.log") 2>&1

resolved_artifacts=$(.venv/bin/rig settings | awk '$1 == "artifacts_path_resolved" {print $2}')
test "${resolved_artifacts}" = "${artifact_root}"

.venv/bin/rig doctor \
  --cluster v4-32 \
  --profile dev \
  --require-tpu \
  --quick \
  --color never
.venv/bin/rig doctor \
  --cluster v4-32 \
  --profile dev \
  --require-tpu \
  --color never

run_cell() {
  local name=$1
  local point=$2
  local seed=$3
  local beta=$4

  if ((${#name} > 40)); then
    printf 'ERROR: run name exceeds the harness 40-character limit: %s\n' \
      "${name}" >&2
    return 1
  fi

  shopt -s nullglob
  local completed=("${artifact_root}"/*-"${name}"-*/result.json)
  local attempts=("${artifact_root}"/*-"${name}"-*)
  shopt -u nullglob
  if ((${#completed[@]} == 1)); then
    .venv/bin/rig verify "$(dirname "${completed[0]}")"
    printf '>>> already complete; skipping %s\n' "${name}"
    return
  fi
  if ((${#completed[@]} > 1)); then
    printf 'ERROR: duplicate completed runs for %s\n' "${name}" >&2
    return 1
  fi
  if ((${#attempts[@]} > 0)); then
    printf 'ERROR: incomplete prior attempt blocks automatic retry for %s\n' \
      "${name}" >&2
    return 1
  fi

  local free_kib
  free_kib=$(df -Pk "${artifact_root}" | awk 'NR == 2 {print $4}')
  if ((free_kib < 524288)); then
    printf 'ERROR: less than 512 MiB free before %s\n' "${name}" >&2
    return 1
  fi

  local command=(
    .venv/bin/rig run fuzzy_topk_reconstruction
    --cluster v4-32
    --profile dev
    --tier 125m
    --context 8k
    --batch-size 16
    --base-learning-rate 0.00390625
    --seed "${seed}"
    --timeout 14400
    --checkpoint-policy none
    --color never
    --name "${name}"
    --study-id "${study_id}"
    --study-point "${point}"
    --study-suite-sha256 "${suite_sha}"
    --
    --sparse-layers 11
    --sparse-mlp-mult 16
    --sparse-top-k 2560
    --sparse-training-steps 4656
    --sparse-mlp-backend choicewise
    --sparsity-diagnostics-every 100
    --reconstruction-coefficient "${beta}"
    --fuzzy-auxk-mode none
    --fuzzy-auxk-coefficient 0
    --fuzzy-auxk-width-ratio 0.5
    --fuzzy-dead-tokens-threshold 10000000
  )

  printf '>>> '
  printf '%q ' "${command[@]}"
  printf '\n'
  "${command[@]}"

  shopt -s nullglob
  local fresh_results=("${artifact_root}"/*-"${name}"-*/result.json)
  shopt -u nullglob
  if ((${#fresh_results[@]} != 1)); then
    printf 'ERROR: successful run did not produce exactly one result for %s\n' \
      "${name}" >&2
    return 1
  fi
  local run_dir
  run_dir=$(dirname "${fresh_results[0]}")
  .venv/bin/rig verify "${run_dir}"

  local run_id=${run_dir##*/}
  case "${run_id}" in
    *-"${name}"-*) ;;
    *)
      printf 'ERROR: refusing cache cleanup for unexpected run id %s\n' \
        "${run_id}" >&2
      return 1
      ;;
  esac
  local cache_path=/tmp/rig-jax-cache-${run_id}
  if [[ -d "${cache_path}" ]]; then
    rm -rf -- "${cache_path}"
  fi
}

# Latin-square coefficient order across the three paired seeds. The existing
# beta=0 and beta=1 anchors are deliberately absent from this queue.
run_cell 125m-v4-recon-beta1of4-s1337 beta-1of4-s1337 1337 0.25
run_cell 125m-v4-recon-beta1of16-s1337 beta-1of16-s1337 1337 0.0625
run_cell 125m-v4-recon-beta1of64-s1337 beta-1of64-s1337 1337 0.015625

run_cell 125m-v4-recon-beta1of16-s1338 beta-1of16-s1338 1338 0.0625
run_cell 125m-v4-recon-beta1of64-s1338 beta-1of64-s1338 1338 0.015625
run_cell 125m-v4-recon-beta1of4-s1338 beta-1of4-s1338 1338 0.25

run_cell 125m-v4-recon-beta1of64-s1339 beta-1of64-s1339 1339 0.015625
run_cell 125m-v4-recon-beta1of4-s1339 beta-1of4-s1339 1339 0.25
run_cell 125m-v4-recon-beta1of16-s1339 beta-1of16-s1339 1339 0.0625

for name in \
  125m-v4-recon-beta1of4-s1337 \
  125m-v4-recon-beta1of16-s1337 \
  125m-v4-recon-beta1of64-s1337 \
  125m-v4-recon-beta1of16-s1338 \
  125m-v4-recon-beta1of64-s1338 \
  125m-v4-recon-beta1of4-s1338 \
  125m-v4-recon-beta1of64-s1339 \
  125m-v4-recon-beta1of4-s1339 \
  125m-v4-recon-beta1of16-s1339; do
  shopt -s nullglob
  results=("${artifact_root}"/*-"${name}"-*/result.json)
  shopt -u nullglob
  test "${#results[@]}" -eq 1
  .venv/bin/rig verify "$(dirname "${results[0]}")"
done
