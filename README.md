# GPT TPU Rig

A simple GPT pretraining rig for Cloud TPU slices, from a single host to
larger multi-host slices. Every algorithm is a polished JAX entry program named
`train.py`, with standalone `config.yaml`, `dev.yaml`, and `smoke.yaml`
documents; shared code handles
reproducible data, machine checks, run capture, protocol validation, and
leaderboards.

Just copy the `recipes/reference` and start hacking.

This is largely inspired by nano-GPT speedrun. The baseline is a GPT-2 with slightly modernized architecture choices (RoPE, GELU, etc).
For hyperparameter transfer, the family implements an opinionated
**fixed-TPP CompleteP hybrid** assembled from two papers:

1. [CompleteP](https://arxiv.org/abs/2505.01618) extends muP with depth scaling for
   the pre-LN transformer (α=1 for L^{-α}), plus AdamW ϵ, weight decay, residual
   block, and embedding rules.
2. [Complete(d)P](https://arxiv.org/abs/2512.22382) adds batch size and token
   duration and corrects the input-embedding AdamW epsilon and unembedding
   parameterization. This repository adopts those two tensor corrections and
   recipe-local batch/data factors, but it does **not** apply a cross-horizon
   `TPP / TPP₀` multiplier. Each fixed-TPP ladder is reanchored independently.

The measured base-LR optimum is `2^-8` and the measured batch optimum is 128
across the 60M–250M 5-TPP ladder, with the same neighborhood observed at 500M
and 20 TPP. A 42-run
[60M/125M ablation](docs/reports/duration-ablation.md) found no benefit from
adding the cross-horizon factor to this recipe. Those results support the
reanchored setup; they do not establish or refute the full Complete(d)P
prescription outside it. The exact contract and limitations are in
[docs/COMPLETEP.md](docs/COMPLETEP.md).

## Start here

The top-level Makefile is the user interface and a set of copyable command
templates:

```bash
make check
make prepare
make run
make baseline
make profile
make report
```

| Target | Purpose |
|---|---|
| `make check` | run every CPU-only check: tests, CLI surface, recipe parsing, report build |
| `make prepare` | synchronize the frozen uv environment, then open the interactive setup wizard |
| `make run` | verify every configured TPU VM and its data, then run the 125M reference tier using the saved profile |
| `make baseline` | compatibility alias for `make run` |
| `make run TARGET=name TIER=250m` | run one tier from the model family in `recipes/name/` |
| `make profile` | run a distributed, validation-free 100-step diagnostic, capture worker 0 steps 11–20, then serve XProf on port 8791 |
| `make report` | integrity-check completed runs and rebuild the standalone `report.html` dashboard |

There is no hosted CI. `make check` is the gate instead — run it before
committing and before spending TPU time. It needs no accelerator and takes
about a minute; `tests/conftest.py` pins `JAX_PLATFORMS=cpu` for the whole
suite, which also stops it from wedging on a multi-host slice that a single
process cannot initialize.

`make prepare` runs these two commands:

```bash
uv --cache-dir /tmp/uv-cache sync --frozen --group dev
uv --cache-dir /tmp/uv-cache run --frozen --no-sync rig prepare \
  --dataset 8B --train-shards 79
```

It asks for the data-cache root, the execution type for this preparation, the
default run type, persistent artifact directory, TPU VM host count, checkpoint
policy, colors, a smoke/development loss target, and an explicit immutable
corpus plus shard prefix. The preparation type is transient; the selected
corpus is saved under `[data]` and applies to every non-smoke run. Data identity
never depends on a training-horizon number: `8B` means the checked-in 8B
manifest, and 79 means its complete train split.
Training duration is independently specified by each family configuration
(20 TPP in the reference official configuration) or by an explicit research
study.
The wizard can then probe JAX/TPU health and prepare the selected dataset;
only the explicit CLI flag `--no-download` skips data preparation. Personal
choices are stored in the gitignored `.rig.toml`; official constants
remain versioned in Git. The personal loss target applies only to
smoke/development work; the official target is fixed at 3.28 and may only be
tightened explicitly.

`make run` requires the saved file to contain a default execution type;
otherwise it stops and asks for `make prepare`. `TARGET` defaults to
`reference`. A custom
target must be a folder beneath `recipes/` containing regular, non-symlink
`train.py`, `config.yaml`, `dev.yaml`, and `smoke.yaml` files. New candidates
use schema 6 family configs
with 60M, 125M, 250M, 500M, and 1B tiers; `TIER` defaults to `125m`.

For a multi-host run using the conventional `shm` cache, preparation requires
`/dev/shm` to be a writable `tmpfs` or `ramfs` on every VM and creates
`shm -> /dev/shm/.speedrun-cache` in each checkout. That directory keeps its
pre-rename name deliberately: renaming it would orphan the prepared corpus
already installed on every TPU VM. It uses `sudo -n` to make
that dedicated directory and completed entries root-owned but writable by the
caller's primary group. This is necessary because systemd-logind defaults to
`RemoveIPC=yes` and recursively removes user-owned `/dev/shm` entries after the
last SSH session for that user ends. Rig does not alter that host-wide
policy. If any VM lacks the mount or non-interactive cache ownership access,
preparation stops with a short instruction before downloading data:

```bash
uv run --frozen --no-sync rig prepare --path shm/ --profile official
```

The cache path is always explicit. The tool supports symlinks, resumable
downloads, free-space checks, and exact header/length/SHA-256 validation. It
never deletes unrelated files. Because `/dev/shm` is ephemeral, preparation may
need to restore the data after a reboot.

### Multi-host TPU slices

A Cloud TPU v4-32 is one slice spread across four TPU VMs, with four TPU v4
chips attached to each VM. It is not four independent v4-8 slices. Run
`make prepare` on worker 0, answer `4` for the TPU VM host count, and accept the
inferred expression when the hostnames follow Cloud TPU's usual convention:

```text
t1v-n-a09f5679-w-[0-3]
```

The expression is ordinary `pdsh` host-list syntax; a comma-separated explicit
list also works. The controller needs `pdsh`, `rsync`, and non-interactive SSH to
itself and every peer. Rig tests that access, but it never creates, copies,
or modifies SSH keys. If the probe fails, add this controller's public key to the
same user's `~/.ssh/authorized_keys` on the TPU VMs, verify
`pdsh -R ssh -w HOSTS hostname`, and rerun preparation.

After the SSH and RAM-cache probes succeed, preparation attempts a
non-interactive `apt-get` installation wherever `rsync` is missing. It then
incrementally mirrors the current checkout—including dirty and untracked
experiment files, but excluding Git metadata, the virtual environment,
`shm/`, data, caches, profiles, and run artifacts—to the same absolute path on
every peer. The mirror is authoritative: a file you delete or rename in the
checkout is removed from the peers on the next synchronization, so stale
modules cannot linger there and shadow current code. Excluded paths are never
deleted, and removal is deferred until the transfer succeeds, so an interrupted
synchronization cannot strip a peer. Keep working notes outside the checkout or
in Git; an untracked file that exists only on a peer will not survive.
It installs `uv` there if needed, synchronizes the frozen
environment, then uses one `pdsh` launch to prepare the selected dataset and
validations concurrently in every VM's local protected RAM cache. Before each
worker's SSH command exits, it protects the completed entries from logout
cleanup. `make run` repeats the source synchronization and launches the trainer
on all configured hosts.

Every trainer process calls `jax.distributed.initialize()` before its first
device query. JAX's runtime rank is `jax.process_index()`; there is no launcher
`RANK` variable to trust. Training and evaluation loaders produce a distinct
rank-local portion of each global batch, while JAX arrays span the whole TPU
mesh. Cloud TPU's JAX rank order need not match the `-w-N` hostname suffix, so
the launcher explicitly marks its controller hostname; only that VM emits
human logs, checkpoints, metrics, and the final result consumed by the harness.

This path targets a single multi-host TPU slice such as v4-32. Cloud TPU
Multislice (multiple separately provisioned slices connected over DCN) also
needs a hybrid ICI/DCN mesh and is a distinct scaling mode.

For automation on a four-host v4-32, bypass the questions:

```bash
uv run --frozen --no-sync rig prepare \
  --non-interactive --path shm/ --profile official \
  --run-profile official --checkpoint-policy qualifying \
  --dataset 8B --train-shards 79 \
  --tpu-vm-count 4 --tpu-vm-hosts 't1v-n-a09f5679-w-[0-3]'
```

### Named corpora

`rig dataset` prepares and distributes corpora by name, independently of any
saved settings:

```bash
rig dataset list                        # capacities, and what is present here
rig dataset prepare hero --shards 105   # download and verify only what you need
rig dataset ship hero --cluster v6e-8   # send it to hosts that cannot download
rig dataset status --cluster v6e-8      # presence here and on the cluster
```

`--shards` matters: 500M at 20 TPP needs 101 of hero's 749 shards, about
19 GiB rather than 140 GiB. Shipping is idempotent, so an interrupted transfer
resumes rather than failing.

Corpus choice is independent of the active cluster:

```toml
[data]
dataset = "hero"
train_shards = 105
```

The name and optional shard prefix are authoritative across clusters. A missing
corpus stops the run with the commands that fix it; there is no capacity-based
fallback and no silent switch to the classic corpus.

### Orchestrating a slice you are not part of

Hosts that are reachable by SSH but have no internet — preemptible pods, for
instance — can be driven from a machine that holds no accelerator:

```toml
[cluster.v6e-8]
tpu_vm_hosts      = "10.164.15.196"
tpu_vm_count      = 1
accelerator       = "TPU v6 lite"
chips_per_host    = 8
remote_controller = true
artifact_host     = "10.164.15.196"   # optional; defaults to the first host
```

`remote_controller` says this machine orchestrates without taking a JAX rank.
One host is designated to write run artifacts, which are pulled back here when
the run ends and periodically while it runs, so a job that dies still leaves a
usable partial curve. A single remote host is a valid target: it launches
remotely but takes no distributed initialization.

For a peer with no route to PyPI, `rig prepare` ships what it needs from here
rather than expecting it to download:

```bash
rig prepare --cluster v6e-8 --non-interactive --yes --offline --ship-data
```

That sends the uv binary, the uv-managed interpreter, the package cache, the
source tree, and (with `--ship-data`) the prepared corpus. `--offline` matters:
`--frozen` alone still consults the package index, so the cache is not enough
on its own.

### Checkpoints

One flag decides what happens to model weights:

```bash
rig run reference --checkpoint-policy always      # keep them
rig run reference --checkpoint-policy qualifying  # keep only at or below target
rig run reference --checkpoint-policy none        # never write them
```

`none` skips the write rather than writing and deleting, which matters at 500M
where a checkpoint is about 2 GB. It is restricted to development research runs.

## Execution types and data

All execution types use one well-defined token format; downloading and tokenization
are never timed.

| Execution type | Data | Intended use |
|---|---|---|
| `smoke` | generated locally | CPU end-to-end wiring checks |
| `dev` | selected named corpus | 5-TPP research runs and diagnostics |
| `official` | selected named corpus plus Fresh10 | 20-TPP confirmation runs |

Non-smoke data is selected directly:

| Corpus | Full training split | Cache root |
|---|---:|---|
| `classic` | 900M tokens | `<data-path>/` |
| `2B` | 1.9B tokens | `<data-path>/fineweb-scaled/2B/` |
| `4B` | 3.9B tokens | `<data-path>/fineweb-scaled/4B/` |
| `8B` | 7.9B tokens | `<data-path>/fineweb-scaled/8B/` |
| `hero` | 74.9B tokens | `<data-path>/fineweb-scaled/hero/` |

`train_shards` may select an explicit prefix of a named manifest. Preparation,
doctor, profiling, and every non-smoke run use that same selection. Scaled
preparation is fail-closed and
starts working only after the corresponding immutable, URL-bearing publication
manifest is checked into `data/manifests/fineweb-scaled-gpt2/`; no placeholder
manifest is accepted. `--check-only` verifies the same manifest and dedicated
folder without mutation.

Each named corpus supplies its own validation file; training never substitutes
the classic validation split for a scaled corpus or routes a validation entry
into the training file list. The `2B`, `4B`, `8B`, and `hero` corpora are nested
prefixes and intentionally share one document-disjoint 100M-token validation
shard. Official evaluation scores exactly its first 10,485,760 predictions.
The harness binds that coverage to the recipe plan before launch; smoke and
development execution types deliberately use shorter diagnostic evaluation.
Document-disjoint means the boundary document is removed from training; it is
not a claim of fuzzy or semantic near-duplicate decontamination.
The classic corpus is the pinned `kjj0/fineweb10B-gpt2` revision used by
[Modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt); scaled corpus
provenance is separate. See [data/README.md](data/README.md) for both contracts.

## Run an algorithm

```bash
# Reference workflow: 125M by default, 20 TPP in the official profile
make run

# Select another size tier
make run TIER=250m

# Run a variant from recipes/dense_control/train.py
make run TARGET=dense_control TIER=125m

# Fast end-to-end check
uv run --frozen --no-sync rig prepare --non-interactive \
  --path shm/ --profile smoke --run-profile smoke --no-doctor
uv run --frozen --no-sync rig run reference --profile smoke

# Run the versioned official configuration (compare timings on like hardware)
uv run --frozen --no-sync rig run reference \
  --profile official

# Walk the first 100 steps of a full-horizon run, schedule untouched
uv run --frozen --no-sync rig run reference --profile dev \
  --tokens-per-parameter 5 --stop-after-step 100
```

Raw step and token-horizon overrides are intentionally absent. A diagnostic
uses `--stop-after-step`, which keeps the full horizon's learning-rate schedule,
warmup, and `m_D` and simply stops early. Its curve is the full run's prefix
step for step, and its `run_kind=diagnostic` keeps it out of leaderboards.

Sweeping a hyperparameter across the ladder is deliberately not built in. One
run is one command, so a study is an ordinary shell loop over `rig run` with
the tiers and overrides you care about, and each point lands in `runs/` as a
normal recorded run. The parameterization rules that make a tier's learning
rate comparable across sizes are in
[the fixed-TPP parameterization contract](docs/COMPLETEP.md), and what those sweeps
measured is in [the hyperparameter transfer note](docs/HYPERPARAMETER_TRANSFER.md).

Name your runs. `make run NAME=cosine-floor` folds a label into the run
directory, turning `20260814T212706.356271Z-reference-9ff5c908` into
`20260814T212706.356271Z-reference-cosine-floor-9ff5c908`. Omit it on a
terminal and `rig run` asks, because a month later the timestamp will not tell
you what you were testing; press enter to accept the unnamed default. The label
is lowercased and reduced to alphanumerics and hyphens. Non-interactive
callers — a study loop, a piped shell, a peer worker — are never prompted and
silently stay unnamed, so scripts cannot hang on a question nobody can answer.

The harness creates a unique persistent run directory, captures stdout/stderr,
validates the final result event and any required checkpoint, hashes artifacts,
and appends a JSONL record. Within a comparable cohort, the trainer's synchronized
accelerator time is the leaderboard score; cold process wall time is recorded
separately. Human progress is streamed live while the machine-readable result
remains isolated on stdout.

Every successful reference run also writes `training.riglog` inside its run
directory: one packed record per optimizer step holding loss, learning rate, and
gradient norm, with cumulative tokens and FLOPs derived from the header rather
than stored. Scalars accumulate on the TPU and transfer only after timed
training, so retaining the complete curve does not add a synchronization to
every step. The harness records the artifact's SHA-256 for later collation
across runs. The byte layout, the column schema, and the permanent metric ids
are documented in [docs/RIGLOG_FORMAT.md](docs/RIGLOG_FORMAT.md).

The reference also writes `validation.csv`. On the official profile it probes
the first 1,048,576 validation predictions every 500 optimizer steps by default,
then records the exact canonical validation as its final FineWeb row. The actual
batch count is derived from the selected context and batch size. Fresh10 rows
may follow it. Probe synchronization and evaluation are included in
`train_seconds`; the final canonical evaluation is not. Both training and
evaluation executables compile once on synthetic zero-valued inputs before
timing. Validation cadence and prefix size live in the cloned recipe YAML.
Smoke and development runs do not probe unless their profile enables it.

The reference is intentionally readable. Its v3 model family uses RoPE,
pre-RMSNorm, a 4× GELU MLP, untied embeddings, fixed 64-wide heads, and the
fixed-TPP CompleteP hybrid documented in
[docs/COMPLETEP.md](docs/COMPLETEP.md). The
60M/125M/250M tiers determine candidate-admission trends; 500M and 1B are
larger confirmation and hero tiers. The historical v1
19,073-step calibration on a TPU v4-8 processed exactly **624,984,064**
training tokens in **1,716.01 synchronized seconds** (28m36s, compilation
excluded), sustaining about **364k tokens/s**, **313 analytic TFLOP/s**, and
**28.5% analytic MFU**. It reached FineWeb validation loss **3.75788** and
Fresh10 macro loss **3.95959**. Those measurements remain labeled as v1 rather
than being attributed to the promoted architecture. The old token budget
remains a historical like-hardware calibration, not the training contract for
the new 20-TPP family ladder.

That calibration is a v4-8 result. A v4-32 run is validly recorded with its
16-device/four-process system identity, but its wall-clock score is not a
like-for-like hardware comparison with the original v4-8 number.

### TPU kernel baseline

The reference [`config.yaml`](recipes/reference/config.yaml) pins the custom
trainable Pallas attention with the memory-bounded tiled output loss. It also
preserves the family shapes, fixed-TPP parameterization contract, objective, schedule,
validation cadence, and TPP rule beside
the entry script. `make run` supplies only saved machine/run policy and lets the
trainer read that versioned file. To create a dense control, clone the reference
and change the clone's `attention_backend` field instead of hiding an algorithm
change in a long launch command:

```bash
uv run --frozen --no-sync rig clone reference dense_control
# edit recipes/dense_control/config.yaml
uv run --frozen --no-sync rig run dense_control --profile official
```

The custom attention kernel includes forward, dQ, and dK/dV kernels, uses a
shape-aware static tile plan, and safely pads non-128-aligned sequence lengths.
An exact runtime/source lookup table supplies measured seeds, falling back to a
deterministic shape heuristic. Both are pure functions of the workload key, so
every process in a multi-host job compiles identical tiles without
communicating; measurement is a deliberate offline step, never part of a run.
The
tiled loss streams vocabulary blocks and recomputes them in backward instead of
materializing full logits. Switching loss implementations keeps all 50,304
storage classes by default; reducing `semantic_vocab_size` in a cloned YAML
profile changes the model objective and is not a mere kernel toggle.

The canonical full-step benchmark improved from 93.196 ms to 75.191 ms with
custom attention and dense loss (435.79k tokens/s, +23.9%). The tiled loss was
77.048 ms in that historical shape and is now the memory-safe family default.
Dense attention's completed 3.75788 run remains the historical quality
control until the promoted baseline has completed its full validation. Details,
APIs, numerical checks, and tuning policy are in
[docs/KERNELS.md](docs/KERNELS.md).

### Profiling and reports

`make profile` uses the profile saved by `make prepare` and launches the trainer
on every configured TPU VM, so the measured computation and collectives use the
real global topology. Compilation happens before capture; canonical validation,
Fresh10, checkpointing, and leaderboard recording are disabled. Worker 0 alone
records its host activity and four local TPU chips, including their participation
in global collectives. This avoids trying to merge independent local trace
sessions from four VM filesystems. After capture, worker 0 starts an isolated,
version-pinned XProf viewer at `http://localhost:8791`; Ctrl-C stops it. Override
the output or capture window with `PROFILE_OUTPUT`, `XPROF_START_STEP`, and
`XPROF_STEPS`.

`make report` scans completed folders beneath `runs/`, checks their recorded
artifact hashes when an immutable record is available, and writes one
self-contained static `report.html` with no CDN dependency. A multi-select run
sidebar controls overlays.

Every successful run is plotted, whatever its profile, token budget, or loss,
and all of them start visible. Each run keeps an honest label — `official`,
`diagnostic`, or `smoke` — so a two-step smoke test is never mistaken for a
record attempt, but nothing is hidden: comparing curves is the point. Whether a
run *qualifies* is a separate question, answered by `rig leaderboard` against
the target in [docs/RULES.md](docs/RULES.md).

Structural integrity is still enforced. A result that is malformed,
unsuccessful, or of the wrong schema is reported as skipped with its reason
rather than silently dropped.

One global selector switches every time-series chart
between **equi-FLOP** (the default, using analytic cumulative estimated FLOPs)
and **equi-step**. Official reference runs record `diagnostics.riglog` at step 1,
every 500 steps, and the final step. The report exposes separate
**Gradient / Update / Parameter** buttons, with one chart for each L1/L2 norm,
mean, standard deviation, and centered third/fourth moment. Timeline charts use
the whole-model values; final-snapshot charts show embeddings, every transformer
block, and the final normalization. Because the final parameter point is
post-update, it exactly describes the saved checkpoint even when qualifying-only
retention later removes that file.

The page embeds its data as gzip, base64-encoded, and inflates it on load with
`DecompressionStream`; the payload is about 99% of the file, so this roughly
halves it. That needs Chrome 80+, Firefox 113+, or Safari 16.4+.

Layer snapshots are scrubbable. A step dragger above them selects which recorded
step every per-scope chart shows, defaulting to the last, and its position is
echoed on each timeline as a faint dashed marker so you can see where in
training you are looking. Hovering any timeline draws a crosshair at the same x
on all of them; a single click pins a vertical line there and clicking again
clears it, while double-click still resets the view. Tooltips report both the
optimizer step and the estimated FLOPs whichever axis is selected, so a point of
interest can be located on the dragger. By default every recorded diagnostic
step is kept, so the dragger moves at the granularity the run recorded;
`make report LAYER_SNAPSHOTS=N` thins the step axis to N if you would rather
have a smaller file. Compatible retained checkpoints are used only
as a fallback for runs recorded without diagnostics.

Charts render into bounded, downsampled canvases only after an input or resize
event—there is no animation loop or idle redraw. Hover inspects the nearest
curve. Pointer dragging draws a selection rectangle and rescales both axes to
that rectangle; reset/double-click restores the full extent, and the wheel does
not modify chart axes. Timeline traces can remain raw or use a configurable EMA,
centered moving mean, or centered moving median. Smoothing is display-only: a
faint raw trace remains visible and hover reports the raw sample as well as the
smoothed value. Learning-rate schedules and layer snapshots remain raw. The
expand button opens any chart in a temporary full-panel dialog; Escape closes
it. All code and data remain inside the static HTML.

### Fresh-domain diagnostic

The canonical FineWeb validation loss remains the only qualification metric. A
separate `fresh10` diagnostic covers ten tiny, temporally fresh domains
with exactly 8,192 scored GPT-2 tokens each: science, medicine, software,
history, open-licensed fiction, government, legal, economics, climate, and
education. Source documents
are published after the pinned FineWeb snapshot, license-audited, cleaned by
a versioned deterministic recipe, and frozen by URL/revision plus SHA-256.

`fresh10` reports each domain independently and a macro-average beside FineWeb.
It does not change qualification, and “fresh” means a strong temporal contamination
control rather than a proof that no equivalent passage ever appeared online.
The entry trainer accepts the frozen set with
`--downstream-manifest data/manifests/fresh10.json --downstream-root PATH`.
It reuses one fixed-shape masked evaluation executable, excludes padding and
cross-document targets, and writes the domain rows plus `fresh10_macro` to
`validation.csv`. Downstream data is manifest-only so its document boundaries,
tokenizer, byte sizes, and content hashes remain part of the run contract.

Useful commands:

```bash
uv run --frozen --no-sync rig doctor --require-tpu
uv run --frozen --no-sync rig settings
uv run --frozen --no-sync rig verify RUN_ID
uv run --frozen --no-sync rig leaderboard --profile official
```

Leaderboards render one block per content-addressed cohort. A cohort fixes the
tier, target TPP and rounding rule, selected data and validation prefix,
qualification target, and hardware topology. Recipe, dense-versus-MoE
architecture, optimizer, learning rate, batch size, and seed remain experiment
dimensions. Early-stopped diagnostics are never ranked.

## Create an algorithm

Every entry has the same explicit configuration layout:

```text
recipes/<algorithm>/train.py
recipes/<algorithm>/config.yaml  # official
recipes/<algorithm>/dev.yaml
recipes/<algorithm>/smoke.yaml
```

Clone the current reference without overwriting anything:

```bash
uv run --frozen --no-sync rig clone reference my_experiment
```

The clone copies the entry program and all three configuration documents
byte-for-byte. Keep
the implementation visible in `train.py` and the experiment-defining model,
optimizer, schedule, objective, kernel, and validation settings in the selected
YAML. Runtime locations, run
identity, seed, and profiling destinations remain command-line concerns.
Fundamental shared data/protocol/UI utilities are welcome when they make entries
shorter and easier to compare.

Each YAML is schema-versioned and complete: `config.yaml` is official,
`dev.yaml` is development, and `smoke.yaml` is the CPU smoke configuration.
`--profile` only selects one of these files; there is no inheritance, overlay,
or unselected profile block. The trainer resolves the selected document
relative to its own file—not the caller's working directory—and rejects
duplicate/unknown keys, unsafe YAML
features, type/range errors, symlinks, and attempts to replace static settings
with hidden launch flags. The harness records both the source SHA-256 and the
fully resolved profile. The small public research surface—tier, context preset,
TPP, base learning rate, batch size, and diagnostic stop point—is recorded
explicitly. A cloned recipe may own additional typed arguments without adding
them to the global harness. Put them after an explicit `--` boundary; `rig`
passes the same opaque tail to plan resolution and execution, and the immutable
run record retains the resulting trainer command:

```bash
rig run my_experiment --profile dev -- --my-recipe-option value
rig profile my_experiment --output-dir profiles/test -- --my-recipe-option value
rig run my_experiment -- --help
```

Arguments before the boundary remain harness-owned, and recipe-local arguments
cannot override the shared execution/data protocol. Fold settled defaults into
the recipe YAML before publishing a stable result.

## Comparable cohorts

A full run receives a content-addressed cohort identity derived from the tier
and parameter anchor, TPP horizon, selected immutable data, validation
contract, target loss, profile, and accelerator topology. Only qualifying runs
inside the same cohort are ranked together, by synchronized training time.
Architecture, optimizer, schedule, precision, batch, learning rate, seed,
sharding, and kernels remain free experiment dimensions. Diagnostics and smoke
runs are never ranked, and records without an explicit cohort are not silently
mixed. There is no composite score; parameter count, FLOP estimates,
throughput, MFU estimate, compilation time, tokens, and loss remain diagnostics.

The full timing, qualification, checkpoint, and human-review rules are in
[docs/RULES.md](docs/RULES.md).

## Repository map

```text
data/manifests/          pinned datasets and hashes
rig/                     CLI, wizard, doctor, data preparation, report
rig/harness/             execution, result protocol, records, scoring
rig/kernels/             shared TPU attention, loss, and autotuning primitives
recipes/reference/       JAX entry trainer + versioned experiment config
tests/                   CPU-only infrastructure tests
runs/                    gitignored persistent run artifacts
```

The project is licensed under Apache-2.0.

[1] Bordelon, B., Chaudhry, H., & Pehlevan, C. (2024). Infinite Limits of Multi-head Transformer Dynamics. In *Advances in Neural Information Processing Systems 37 (NeurIPS 2024)*. arXiv:2405.15712.
