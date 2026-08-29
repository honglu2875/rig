#!/usr/bin/env bash
set -euo pipefail

study_root=${RIG_RECON_STUDY_ROOT:-/tmp/GPT-speedrun-TPU-fuzzy-reconstruction-auxk}
artifact_root=${RIG_RECON_RUNS_ROOT:-${study_root}/runs}
study_id=fuzzy-topk-reconstruction-auxk-125m-v4-3seed
suite_path=recipes/fuzzy_topk_reconstruction/study-suite-125m-v4-reconstruction-auxk-3seed.json

cd "${study_root}"
test -z "$(git status --porcelain --untracked-files=no)"
suite_sha=$(sha256sum "${suite_path}" | cut -d' ' -f1)
mkdir -p "${artifact_root}"
exec > >(tee -a "${artifact_root}/queue.log") 2>&1

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
  local recipe=$1
  local name=$2
  local point=$3
  local seed=$4
  local stop_after=$5

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
    printf 'ERROR: incomplete prior attempt blocks automatic retry for %s\n' "${name}" >&2
    return 1
  fi

  local command=(
    .venv/bin/rig run "${recipe}"
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
  )
  if [[ "${stop_after}" != "full" ]]; then
    command+=(--stop-after-step "${stop_after}")
  fi
  command+=(
    --
    --sparse-layers 11
    --sparse-mlp-mult 16
    --sparse-top-k 2560
    --sparse-training-steps 4656
    --sparse-mlp-backend choicewise
    --sparsity-diagnostics-every 100
    --reconstruction-coefficient 1.0
    --fuzzy-auxk-width-ratio 0.5
    --fuzzy-dead-tokens-threshold 10000000
  )
  if [[ "${recipe}" == "fuzzy_topk_reconstruction_auxk" ]]; then
    command+=(--fuzzy-auxk-mode auxk --fuzzy-auxk-coefficient 0.03125)
  else
    command+=(--fuzzy-auxk-mode none --fuzzy-auxk-coefficient 0)
  fi

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

  # A run gets a unique XLA cache, so a verified result cannot reuse it. Keep
  # failed-run caches for debugging, but reclaim successful ones before the
  # next cell so persistent rigvec logs cannot exhaust the controller disk.
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

# Systems-only seed 1350 gates. These are not endpoint evidence.
run_cell \
  fuzzy_topk_reconstruction \
  125m-v4-reconstruction-gate-v2-s1350 \
  gate-v2-reconstruction-s1350 \
  1350 \
  120
run_cell \
  fuzzy_topk_reconstruction_auxk \
  125m-v4-reconstruction-auxk-gate-v2-s1350 \
  gate-v2-reconstruction-auxk-s1350 \
  1350 \
  120
RIG_RECON_RUNS_ROOT="${artifact_root}" \
  .venv/bin/python recipes/fuzzy_topk_reconstruction/validate_v4_125m_gates.py

# Six missing endpoints. The three sealed parent cells are deliberately absent.
run_cell fuzzy_topk_reconstruction \
  125m-v4-reconstruction-full-s1337 reconstruction-s1337 1337 full
run_cell fuzzy_topk_reconstruction_auxk \
  125m-v4-reconstruction-auxk-full-s1337 reconstruction-auxk-s1337 1337 full
run_cell fuzzy_topk_reconstruction_auxk \
  125m-v4-reconstruction-auxk-full-s1338 reconstruction-auxk-s1338 1338 full
run_cell fuzzy_topk_reconstruction \
  125m-v4-reconstruction-full-s1338 reconstruction-s1338 1338 full
run_cell fuzzy_topk_reconstruction \
  125m-v4-reconstruction-full-s1339 reconstruction-s1339 1339 full
run_cell fuzzy_topk_reconstruction_auxk \
  125m-v4-reconstruction-auxk-full-s1339 reconstruction-auxk-s1339 1339 full

for name in \
  125m-v4-reconstruction-full-s1337 \
  125m-v4-reconstruction-auxk-full-s1337 \
  125m-v4-reconstruction-auxk-full-s1338 \
  125m-v4-reconstruction-full-s1338 \
  125m-v4-reconstruction-full-s1339 \
  125m-v4-reconstruction-auxk-full-s1339; do
  shopt -s nullglob
  results=("${artifact_root}"/*-"${name}"-*/result.json)
  shopt -u nullglob
  test "${#results[@]}" -eq 1
  .venv/bin/rig verify "$(dirname "${results[0]}")"
done
