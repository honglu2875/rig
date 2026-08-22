---
name: run-study
description: Set up an isolated rig clone or research branch, define a controlled pretraining study, validate it, and launch or monitor TPU runs. Use when asked to fork a recipe, plan or queue an ablation, prepare a sibling clone, run on v4-32, or resume an experiment without disturbing an active checkout.
---

# Run a study

Work from the repository root. The usual active checkout is
`/home/cubic27/GPT-speedrun-TPU`; use a sibling such as
`/home/cubic27/GPT-speedrun-TPU-<study>` when another branch or run must remain
undisturbed.

## Isolate the work

1. Inspect `git status --short`, the current branch, and active processes before
   changing anything. Untracked files belong to the user.
2. For a sibling clone, run `git clone git@github.com:honglu2875/rig.git
   /home/cubic27/GPT-speedrun-TPU-<study>`, then create
   `research/<study>` from current `origin/main`. For an existing sibling,
   fetch, switch to `main`, fast-forward it, then create the branch.
3. Run `uv --cache-dir /tmp/uv-cache sync --frozen --group dev`. `.rig.toml`,
   `.env.hf`, `shm/`, and `runs/` are local state, not source. Initialize the
   clone with `rig prepare`, or deliberately reuse the trusted machine settings
   and prepared corpus; never commit or print secrets.
4. Keep the active training clone stable while a queue runs. Develop in the
   sibling, push the branch, merge only after review, then pull into the active
   clone between runs.

## Define the experiment

- Prefer `rig clone <source> <new_recipe>`; for MoE work, fork
  `reference_moe`. It creates `recipes/<new_recipe>/{train.py,config.yaml,
  dev.yaml,smoke.yaml}`. These three YAML files are complete, flat documents:
  `config.yaml` is official, `dev.yaml` is research/development, and
  `smoke.yaml` is the CPU smoke contract.
- Keep architecture, optimizer, and algorithm changes visible in recipe-local
  code/config. Put generic data, protocol, diagnostics, and display utilities
  in `rig/`. Do not hide scientific changes in shared helpers.
- Expose temporary typed recipe options only after the explicit harness
  boundary: `rig run <recipe> ... -- --recipe-option value`. Once a treatment
  settles, fold its defaults into all appropriate YAMLs.
- Write the comparison table before running: baseline, treatment coordinates,
  tier, context, TPP, global batch, base LR, seed(s), dataset/validation split,
  cluster, checkpoint policy, and stopping rule. Reuse a baseline only when
  source/config, data order, seed, topology, and all scientific coordinates
  match—not because its name looks similar.
- Choose a smallest informative grid. Use one seed for mechanism/prototype
  rejection; add seeds only after the treatment survives. Estimate time from a
  short faithful run before committing expensive 250M/500M work.

Example: `research/expert-load-scaling` cloned `reference_moe` into
`recipes/expert_load_moe`, held seed 1350 and the 125M/8k/5-TPP recipe fixed,
and varied only `{gradient,update} × {0.5,1}`.

## Validate before TPU time

1. Add focused unit tests for parsing, invariants, and the changed math. For a
   fork, test that its neutral setting is equivalent to the source when that is
   part of the design.
2. Run `make check`. It executes tests, Ruff, every dev plan, every CPU smoke
   run, and a report build.
3. Inspect the exact JSON plan directly when needed:
   `JAX_PLATFORMS=cpu .venv/bin/python recipes/<recipe>/train.py --profile dev
   --print-plan -- <recipe-options>`.
4. Check the named machine explicitly; never trust an ambient default:
   `.venv/bin/rig doctor --cluster v4-32 --profile dev --require-tpu --quick`,
   then repeat without `--quick` before the queue so topology, collectives, and
   data are checked.
5. Commit and push the exact branch before full runs. Record the clean HEAD and
   inspect `git status --short` again. Editor swap files can make provenance
   dirty even when training code is unchanged; close the editor or keep its
   temporary files outside `recipes/`.

## Launch and monitor

Use explicit flags and exact run names. For the expert-load example:

```bash
.venv/bin/rig run expert_load_moe --context 8k --cluster v4-32 --profile dev \
  --tier 125m --tokens-per-parameter 5 --batch-size 16 \
  --base-learning-rate 0.00390625 --seed 1350 \
  --checkpoint-policy none --name 125m-load-gradient-c0p5-s1350 -- \
  --expert-load-scaling-mode gradient --expert-load-scaling-strength 0.5
```

Put a sequential sweep in `/tmp/<study>.sh` and launch it in a named tmux
session/window so the user can observe it:

```bash
tmux new-session -d -s rig-<study> -n sweep "bash /tmp/<study>.sh"
tmux attach -t rig-<study>
```

Use `set -euo pipefail` in the queue script, one run at a time on a single TPU
slice, explicit `--cluster v4-32`, and a realistic `--timeout`. Print each
command before launching it. Do not silently retry a failed scientific run.

Watch the first compilation and optimizer steps, then check `stdout.log`,
`stderr.log`, and device utilization periodically. A completed run under
`runs/<timestamp>-<recipe>-<name>-<id>/` must have `result.json` with
`status: ok`, `training.riglog`, `diagnostics.riglog`, `validation.csv`, and
`metrics.json`. Run `.venv/bin/rig verify <run-id-or-path>` before analysis.
Report progress with finished/active/pending counts, current step and
throughput, failures, and `tmux attach -t rig-<study>`.

Do not use a different TPU type merely because it is available: topology is
part of the numerical result. Do not cancel jobs, push branches, or reserve new
machines unless the user authorized those state changes.
