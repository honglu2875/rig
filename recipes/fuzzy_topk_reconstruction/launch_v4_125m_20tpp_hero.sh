#!/usr/bin/env bash
set -euo pipefail

study_root=${RIG_RECON_HERO_STUDY_ROOT:-/tmp/GPT-speedrun-TPU-fuzzy-reconstruction-auxk}
artifact_root=${RIG_RECON_HERO_RUNS_ROOT:-${study_root}/runs}
study_id=fuzzy-topk-reconstruction-20tpp-hero-125m-v4
suite_path=recipes/fuzzy_topk_reconstruction/study-suite-125m-v4-reconstruction-20tpp-hero.json
source_queue_session=rig-v4-reconstruction-beta-sweep

cd "${study_root}"
test -z "$(git status --porcelain --untracked-files=no)"
suite_sha=$(sha256sum "${suite_path}" | cut -d' ' -f1)
selector_commit=$(git rev-parse HEAD)
mkdir -p "${artifact_root}"
exec > >(tee -a "${artifact_root}/hero-20tpp-queue.log") 2>&1

resolved_artifacts=$(.venv/bin/rig settings | awk '$1 == "artifacts_path_resolved" {print $2}')
test "${resolved_artifacts}" = "${artifact_root}"

expected_sweep_names=(
  125m-v4-recon-beta1of4-s1337
  125m-v4-recon-beta1of16-s1337
  125m-v4-recon-beta1of64-s1337
  125m-v4-recon-beta1of16-s1338
  125m-v4-recon-beta1of64-s1338
  125m-v4-recon-beta1of4-s1338
  125m-v4-recon-beta1of64-s1339
  125m-v4-recon-beta1of4-s1339
  125m-v4-recon-beta1of16-s1339
)

RESULT_PATH=
resolve_named_result() {
  local name=$1
  shopt -s nullglob
  local matches=("${artifact_root}"/*-"${name}"-*/result.json)
  shopt -u nullglob
  if ((${#matches[@]} > 1)); then
    printf 'ERROR: duplicate completed results for %s\n' "${name}" >&2
    return 1
  fi
  RESULT_PATH=${matches[0]:-}
}

while true; do
  completed=0
  for name in "${expected_sweep_names[@]}"; do
    resolve_named_result "${name}"
    if [[ -n "${RESULT_PATH}" ]]; then
      completed=$((completed + 1))
    fi
  done
  if ((completed == ${#expected_sweep_names[@]})); then
    break
  fi

  pane_state=$(tmux list-panes -t "${source_queue_session}" \
    -F '#{pane_dead} #{pane_dead_status}' 2>/dev/null || true)
  if [[ -z "${pane_state}" ]]; then
    printf 'ERROR: source queue disappeared with only %d/%d endpoints complete\n' \
      "${completed}" "${#expected_sweep_names[@]}" >&2
    exit 1
  fi
  if [[ "${pane_state}" == 1\ * ]]; then
    printf 'ERROR: source queue exited with state %s and only %d/%d endpoints complete\n' \
      "${pane_state}" "${completed}" "${#expected_sweep_names[@]}" >&2
    exit 1
  fi
  printf '>>> waiting for coefficient sweep: %d/%d complete\n' \
    "${completed}" "${#expected_sweep_names[@]}"
  sleep 60
done

declare -a new_results=()
for name in "${expected_sweep_names[@]}"; do
  resolve_named_result "${name}"
  .venv/bin/rig verify "$(dirname "${RESULT_PATH}")"
  new_results+=("${RESULT_PATH}")
done

beta_one_results=(
  "${artifact_root}/20260829T172612.351718Z-fuzzy_topk_reconstruction-125m-v4-reconstruction-full-s1337-8f811907/result.json"
  "${artifact_root}/20260829T184359.087601Z-fuzzy_topk_reconstruction-125m-v4-reconstruction-full-s1338-f934ed13/result.json"
  "${artifact_root}/20260829T190808.764562Z-fuzzy_topk_reconstruction-125m-v4-reconstruction-full-s1339-b813960f/result.json"
)

records=$(mktemp /tmp/fuzzy-reconstruction-hero-records.XXXXXX)
selection_tmp=$(mktemp /tmp/fuzzy-reconstruction-hero-selection.XXXXXX)
trap 'rm -f -- "${records}" "${selection_tmp}"' EXIT

append_candidate() {
  local path=$1
  local expected_beta=$2
  local expected_seed=$3
  test -f "${path}"
  .venv/bin/rig verify "$(dirname "${path}")"
  jq -e \
    --argjson beta "${expected_beta}" \
    --argjson seed "${expected_seed}" \
    '.status == "ok"
      and .seed == $seed
      and .implementation.reconstruction.coefficient == $beta
      and .metrics.training_steps == 4656
      and (.metrics.validation_loss | type == "number")
      and .contract.context_preset == "8k"
      and .contract.model.layers == 11
      and .contract.model.d_model == 640
      and .contract.model.mlp_mult == 16
      and .contract.model.mlp_top_k == 2560' \
    "${path}" >/dev/null
  local run_id
  run_id=$(basename "$(dirname "${path}")")
  jq -r --arg run_id "${run_id}" \
    '[.implementation.reconstruction.coefficient, .seed,
      .metrics.validation_loss, $run_id] | @tsv' \
    "${path}" >>"${records}"
}

for path in "${beta_one_results[@]}"; do
  seed=$(jq -r '.seed' "${path}")
  append_candidate "${path}" 1 "${seed}"
done

for path in "${new_results[@]}"; do
  beta=$(jq -r '.implementation.reconstruction.coefficient' "${path}")
  seed=$(jq -r '.seed' "${path}")
  case "${beta}" in
    0.25|0.0625|0.015625) ;;
    *)
      printf 'ERROR: unexpected lower-beta candidate %s in %s\n' \
        "${beta}" "${path}" >&2
      exit 1
      ;;
  esac
  append_candidate "${path}" "${beta}" "${seed}"
done

jq -Rn \
  --arg study_id "${study_id}" \
  --arg suite_sha256 "${suite_sha}" \
  --arg selector_commit "${selector_commit}" \
  --arg selection_rule "lowest three-seed arithmetic-mean final dev validation loss; exact ties choose the smaller positive reconstruction coefficient" \
  '[inputs | select(length > 0) | split("\t") | {
      coefficient: (.[0] | tonumber),
      seed: (.[1] | tonumber),
      validation_loss: (.[2] | tonumber),
      run_id: .[3]
    }]
   | sort_by(.coefficient, .seed)
   | group_by(.coefficient)
   | map({
       coefficient: .[0].coefficient,
       mean_validation_loss: (map(.validation_loss) | add / length),
       endpoints: map({seed, validation_loss, run_id})
     })
   | sort_by(.mean_validation_loss, .coefficient)
   | {
       schema_version: 1,
       status: "selected",
       study_id: $study_id,
       study_suite_sha256: $suite_sha256,
       selector_commit: $selector_commit,
       selection_rule: $selection_rule,
       selected: .[0],
       candidates: .
     }' <"${records}" >"${selection_tmp}"

jq -e '
  (.candidates | length) == 4
  and (all(.candidates[]; (.endpoints | length) == 3))
  and ([.candidates[].coefficient] | sort) == [0.015625, 0.0625, 0.25, 1]
  and (.selected.coefficient > 0)' "${selection_tmp}" >/dev/null

selection_path=${artifact_root}/fuzzy-topk-reconstruction-20tpp-hero-selection.json
if [[ -e "${selection_path}" ]]; then
  if ! cmp -s "${selection_tmp}" "${selection_path}"; then
    printf 'ERROR: existing hero selection ledger disagrees with recomputation\n' >&2
    exit 1
  fi
else
  mv "${selection_tmp}" "${selection_path}"
fi

selected_beta=$(jq -r '.selected.coefficient' "${selection_path}")
selected_mean=$(jq -r '.selected.mean_validation_loss' "${selection_path}")
case "${selected_beta}" in
  1) selected_tag=beta1 ;;
  0.25) selected_tag=beta1of4 ;;
  0.0625) selected_tag=beta1of16 ;;
  0.015625) selected_tag=beta1of64 ;;
  *)
    printf 'ERROR: selected beta is outside the admitted grid: %s\n' \
      "${selected_beta}" >&2
    exit 1
    ;;
esac
printf '>>> selected reconstruction coefficient %s (%s; mean validation %.9f)\n' \
  "${selected_beta}" "${selected_tag}" "${selected_mean}"

.venv/bin/rig doctor \
  --cluster v4-32 \
  --profile official \
  --require-tpu \
  --quick \
  --color never
.venv/bin/rig doctor \
  --cluster v4-32 \
  --profile official \
  --require-tpu \
  --color never

run_cell() {
  local name=$1
  shift
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
  if ((free_kib < 1048576)); then
    printf 'ERROR: less than 1 GiB free before %s\n' "${name}" >&2
    return 1
  fi

  printf '>>> '
  printf '%q ' "$@"
  printf '\n'
  "$@"

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

dense_name=125m-20tpp-dense-s1350
run_cell "${dense_name}" \
  .venv/bin/rig run reference \
  --cluster v4-32 \
  --profile official \
  --tier 125m \
  --context 8k \
  --tokens-per-parameter 20 \
  --batch-size 16 \
  --base-learning-rate 0.00390625 \
  --seed 1350 \
  --timeout 28800 \
  --checkpoint-policy none \
  --color never \
  --name "${dense_name}" \
  --study-id "${study_id}" \
  --study-point dense-reference-s1350 \
  --study-suite-sha256 "${suite_sha}"

treatment_name=125m-20tpp-recon-${selected_tag}-s1350
run_cell "${treatment_name}" \
  .venv/bin/rig run fuzzy_topk_reconstruction \
  --cluster v4-32 \
  --profile official \
  --tier 125m \
  --context 8k \
  --batch-size 16 \
  --base-learning-rate 0.00390625 \
  --seed 1350 \
  --timeout 28800 \
  --checkpoint-policy none \
  --color never \
  --name "${treatment_name}" \
  --study-id "${study_id}" \
  --study-point "selected-reconstruction-${selected_tag}-s1350" \
  --study-suite-sha256 "${suite_sha}" \
  -- \
  --sparse-layers 11 \
  --sparse-mlp-mult 16 \
  --sparse-top-k 2560 \
  --sparse-training-steps 18624 \
  --sparse-mlp-backend choicewise \
  --sparsity-diagnostics-every 100 \
  --reconstruction-coefficient "${selected_beta}" \
  --fuzzy-auxk-mode none \
  --fuzzy-auxk-coefficient 0 \
  --fuzzy-auxk-width-ratio 0.5 \
  --fuzzy-dead-tokens-threshold 10000000

resolve_named_result "${dense_name}"
jq -e '
  .status == "ok"
  and .seed == 1350
  and .metrics.training_steps == 18838
  and .metrics.tokens_processed == 2469134336
  and .metrics.target_tokens_per_parameter == 20' \
  "${RESULT_PATH}" >/dev/null

resolve_named_result "${treatment_name}"
jq -e \
  --argjson beta "${selected_beta}" \
  '.status == "ok"
    and .seed == 1350
    and .implementation.reconstruction.coefficient == $beta
    and .metrics.training_steps == 18624
    and .metrics.tokens_processed == 2441084928
    and .metrics.sparsity_diagnostics_every == 100' \
  "${RESULT_PATH}" >/dev/null

printf '>>> 20-TPP hero pair complete and verified\n'
