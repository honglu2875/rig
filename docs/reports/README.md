# Reports — what each one shows and how to rebuild it

Every dashboard here, the runs behind it, and the command that reproduces it.
Commands are **demonstrative**: they use the current CLI and reproduce the
*design*, not the exact invocation from the time. Seeds, tiers, and grids are
exact.

## The logs live on HuggingFace

**[huggingface.co/datasets/quintic/rig-logs](https://huggingface.co/datasets/quintic/rig-logs)**
— 528 archived run folders representing 522 unique run IDs across twenty-three
studies, laid out as `<study>/<run-name>/`, at full recorded resolution. Six
controls are intentionally repeated inside the self-contained reconstruction
study. That is the archive of record; its
[dataset card](https://huggingface.co/datasets/quintic/rig-logs/blob/main/README.md)
mirrors this catalog and adds archive and reproduction metadata.

The dashboards committed here are **summaries** of those logs, thinned so they
stay portable. Nothing in them is a substitute for the logs: they are one
rendering at one fidelity, and a thinned curve is indistinguishable on screen
from a complete one. When a number matters, read it from the `.riglog`.

```python
from huggingface_hub import hf_hub_download
from rig import logpack

path = hf_hub_download(
    "quintic/rig-logs",
    "batch-sweep-60M/60m-5tpp-bs128-lr2e-8-s1337/training.riglog",
    repo_type="dataset",
)
log = logpack.read_log(path)
log.series("train_loss")          # every optimizer step
```

## What "summary" means here

Every series is thinned to at most **1,440 points**. Per-layer diagnostic
charts additionally keep a bounded number of step frames — 8 for most studies,
and more for the two where the per-layer behaviour is the subject rather than a
by-product:

| report | curve points | layer frames | size |
|---|--:|--:|--:|
| batch-size-sweep-60M | 1,440 | 400 | 44.3 MB |
| batch-size-sweep-500M | 1,440 | 1,440 | 44.3 MB |
| batch-size-sweep-250M | 1,440 | 8 | 15.4 MB |
| lr-batch-sweep-125M | 1,440 | 8 | 8.2 MB |
| 3-seed-gradient-spike | 1,440 | 8 | 6.6 MB |
| 8k-lr-sweep-60M | 1,440 | 8 | 2.4 MB |
| moe-lr-sweep-8k | 1,440 | 8 | 7.2 MB |
| seed-variance | 1,440 | — | 8.4 MB |
| moe-ablations | 480 bins | — | 0.07 MB |
| expert-load-scaling | exact endpoints | — | 0.02 MB |
| moe-weight-decay | exact endpoints | — | 0.04 MB |
| gumbel-local-moe | exact endpoints + mechanism reductions | — | 0.04 MB |
| sparse-autoencoder-eqflop | exact endpoints + compute derivation | — | 0.02 MB |
| fuzzy-topk-three-arm-ladder | exact endpoints + paired-seed/compute tables | — | 0.03 MB |
| fuzzy-topk-sparsity-diagnostics | exact mechanism reductions + 40.19 GiB raw vectors | — | 0.02 MB |
| fuzzy-topk-dead-latent-rejected-paths | exact endpoints + mechanism decisions | — | 0.01 MB |
| fuzzy-topk-reconstruction-125m | exact endpoints + paired intervals + probe reductions | — | 0.02 MB |

The two large ones carry layer detail because gradient spikes are visible in
it, and studying them is the point. This is deliberate discretion, not a
default: keep it to a couple of files so the repository stays clonable.

Charts resample against the visible span as you zoom, keeping each pixel
bucket's minimum and maximum rather than one representative point — so a spike
inside the embedded data stays visible at every zoom level. It cannot recover a
sample that thinning already dropped.

Charts are per-metric, and a metric no selected run recorded is not drawn at
all — the panel is hidden rather than left as an empty frame. Routed runs
record routing series a dense run never will, so most reports carry charts that
do not apply to part of the selection, and a grid of empty frames would bury
the ones that do.

`seed-variance.html` uses a different reduction: it first computes the
across-seed mean and sample standard deviation at each exact logged step. Each
metric gets two mean ±1 SD panels and two SD-only panels. Mean- and SD-based
LTTB selections are unioned into at most 1,440 retained steps. The selector
exposes 45 training and 288 diagnostic series; 96 expert-indexed load series
are deliberately omitted because expert identities are permutation-symmetric
across seeds. Permutation-invariant router summaries remain available.

`moe-ablations.html` is a static findings report rather than a general run
dashboard. Its endpoint plots use the exact final values. The only reduced
trajectories are the auxiliary-coefficient LM-loss curves: after subtracting
the logged auxiliary term, each is averaged into 480 equal-width FLOP bins.
The first 10% warmup region is omitted from that one chart to resolve the late
separation. The 72 KiB page uses inline SVG and no runtime JavaScript or fetch.

`expert-load-scaling.html` is another static findings report. It uses exact
validation endpoints and permutation-invariant reductions of the per-expert
diagnostics. The 19 KiB page uses inline SVG and no runtime JavaScript or fetch;
the study browser carries the complete trajectories separately.

`moe-weight-decay.html` reports exact validation endpoints for 36 runs, with
three-seed means, sample-SD whiskers, and same-seed differences. Its 40 KiB
page is static inline SVG with no runtime JavaScript or fetch; the study
browser carries the complete trajectories separately.

`gumbel-local-moe.html` is a prose-led mechanism report over eight runs. Its
three figures use exact validation endpoints, recorded cost multipliers,
last-200-step permutation-invariant router summaries, and the first identical
update diagnostic. The 38 KiB page is static inline SVG with no runtime
JavaScript or fetch. The study browser adds mean-across-block timelines for the
four local-objective metrics; raw logs preserve every block separately.

`sparse-autoencoder-eqflop.html` is a static algorithmic study over one dense
anchor and twelve sparse TopK-ReLU treatments. It reports exact endpoints and
the declared forward/backward contraction budget, then shows how depth and
whole optimizer steps make the full-model grid equal-FLOP. The 22 KiB page uses
inline SVG and no runtime JavaScript or fetch. Throughput is included only to
make the current kernel limitation explicit.

`fuzzy-topk-three-arm-ladder.html` is a static paired-seed findings report over
24 complete runs. It compares dense GELU, fixed-group fuzzy TopK, and a
doubly-fuzzy input-plus-hidden selector at matched total active matrix FLOPs.
The page reports every validation endpoint, sample SD, architecture, stored
parameter count, issued FLOPs, throughput, provenance hash, and the explicit
incomplete-250M evidence boundary. It uses inline SVG and no runtime fetch.

`fuzzy-topk-sparsity-diagnostics.html` is a static mechanism report over twelve
fuzzy-only runs from 60M through 500M. It shows when persistent feature
inactivity emerges, its layer profile, and the associated within-group
competition. The study browser carries the one-off interactive distributions:
dead-layer heatmaps, activation-frequency ridgelines, a draggable-step
histogram, and positive-frequency quantiles.

`fuzzy-topk-dead-latent-rejected-paths.html` is a static decision report over
42 new runs split into two rejected paths. Thirty-three balance/homeostasis
runs culminate in a three-seed frequency-floor scale-up; nine tied-init and
zero-forward ghost-AuxK runs end at a 60M mechanism rejection. The page uses
exact endpoints and derived feature-survival reductions, inline SVG, and no
runtime fetch. Parent runs are referenced from the existing sparsity study by
exact run ID rather than archived again.

`fuzzy-topk-reconstruction-125m.html` is a static findings report over 23
archived entries: a three-seed reconstruction-coefficient sweep, literal AuxK,
and a held-out-seed long-horizon pair. It distinguishes the incremental
reconstruction contrast from the combined fuzzy-versus-dense contrast, reports
paired intervals and relative cross-entropy/perplexity effects, and shows the
quality–feature-survival frontier. The page uses exact endpoints and fixed
validation probes, inline SVG, and no runtime fetch.

Which metrics get charted is a declared list in `rig/report.py`, separate from
the metric registry, because how a quantity should be drawn is a judgement the
registry cannot make. Ordinary report metrics remain lines against time or
layer. The fuzzy vector studies are the explicit exception: they precompute
zero-aware log-frequency histograms and positive quantiles from full-neuron
vectors and render them with dedicated histogram, ridgeline, and heatmap views
rather than bending a distribution into a scalar timeline.

## The study browser

[`study-browser.html`](study-browser.html) carries no run data at all — about
80 KB. It
lists the studies, renders each one's card from the dataset, and fetches only
that study's overview (0.05–1.1 MB) when you pick one. Each study always links
to its raw files on Hugging Face. A second, separately labelled click loads the
full report payload generated by the export pipeline and states its size before
it starts: 6.4 MB for the 8k sweep, 138 MB for the 500M one. Nothing downloads
on load.

Every study export publishes both browser views: a compact overview snapshot
and an explicitly loaded `full.json.gz`. Existing studies retain every
recorded point in that expanded payload. The reconstruction study is the first
declared exception: its 5.52 MB expanded view keeps at most 2,048 points per
run-series and 32 layer frames, while its 4.60 GB raw archive remains lossless.
The browser labels that action “expanded,” not “full.” Every report payload the
browser fetches is ordinary JSON, so the page never needs to understand the
packed log format.

The run selector can save either a self-contained HTML report for the visible
runs or a `.tar.gz` containing their original training and diagnostic
`.riglog` files. The latter is available when a report was opened through the
study browser, which knows the corresponding raw-log location. Browsers that
support the File System Access API show a native destination-and-filename
picker; other browsers retain the ordinary download fallback.

The first explicit full-view load also stores the compressed `full.json.gz` in
browser-managed Cache Storage. Reopening the same published study reuses that
copy; a changed published size selects a new cache entry and removes the older
version for that study. The cache is best-effort—the browser may evict it, and
clearing site data removes it—so the Hugging Face copy remains authoritative.

## Hardware is part of a result

The same configuration and seed lands 0.004–0.023 nats apart on a 16-chip
v4 slice versus an 8-chip v6e — the same size as the seed effect. The data is
identical (the stream is invariant under process count, verified) and so is the
attention tile plan; what differs is that gradients reduce across a different
number of devices and each chip holds a different share of the batch.

Every dashboard therefore shows chip kind, chip count, and process count beside
each run, and the run filter matches on chip. The 60M seed cohort is entirely
TPU v6 lite at 1 process and 8 chips; the 125M cohort is entirely TPU v4 at 4
processes and 16 chips. `batch-size-sweep-500M` remains the only individual
study that mixes the two topologies.

Both duration-ablation cohorts are TPU v4 at 4 processes and 16 chips, so their
60M-to-125M comparison does not cross a hardware boundary.

Both MoE ablation studies are also TPU v4 at 4 processes and 16 chips. The
no-bias comparisons are paired within that topology; every coefficient point
uses the same topology and seed.

The expert-load scaling study is likewise entirely TPU v4 at 4 processes and
16 chips. Its baseline and four interventions use the same seed and topology.

The MoE weight-decay study is entirely TPU v4 at 4 processes and 16 chips.
Every decay cell uses seeds 1337–1339 on that same topology.

The Gumbel-local MoE study also uses TPU v4 at 4 processes and 16 chips for all
eight arms. Its K=0/K=2 comparison is paired within that topology.

The sparse-autoencoder study uses TPU v4 at 4 processes and 16 chips for all
13 arms. Hardware is fixed, but wall time is not the comparison coordinate:
the selected-row implementation is memory/latency bound and much slower than
the equal-algorithmic-FLOP dense anchor.

The fuzzy TopK three-arm study also uses TPU v4 at 4 processes and 16 chips for
all 24 archived runs. Its matched coordinate is total active matrix FLOPs;
backend-issued FLOPs and throughput are reported separately. The incomplete
double-fuzzy 250M attempt has no endpoint and is not in the archive.

The fuzzy TopK sparsity-diagnostics study likewise uses TPU v4 at 4 processes
and 16 chips for all twelve runs, spanning 60M through 500M without crossing a
topology boundary.

Both fuzzy TopK dead-latent rejected paths use TPU v4 at 4 processes and 16
chips for all 42 new runs. The balance scale-up is paired to exact controls on
that same topology in the already-sealed sparsity study; the ghost path is a
one-seed 60M mechanism screen.

The fuzzy TopK reconstruction study also stays on TPU v4 at 4 processes and 16
chips. Its short sweep uses three paired seeds; its long-horizon pair uses one
held-out seed. Physical reconstruction work is reported rather than equalized:
the ladder coordinate remains total deployed active matrix FLOPs.

## Contents

| report | runs | tier(s) | what varies | logs |
|---|--:|---|---|---|
| [batch-size-sweep-60M](batch-size-sweep-60M.html) | 75 | 60M | batch × LR × seed | [`batch-sweep-60M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/batch-sweep-60M) |
| [lr-batch-sweep-125M](lr-batch-sweep-125M.html) | 27 | 125M | batch × LR × seed | [`lr-batch-sweep-125M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/lr-batch-sweep-125M) |
| [batch-size-sweep-250M](batch-size-sweep-250M.html) | 36 | 250M | batch × LR × seed | [`batch-sweep-250M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/batch-sweep-250M) |
| [batch-size-sweep-500M](batch-size-sweep-500M.html) | 12 | 500M | batch × LR × seed, 5 and 20 TPP | [`batch-sweep-500M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/batch-sweep-500M) |
| [3-seed-gradient-spike](3-seed-gradient-spike.html) | 12 | 250M | LR × seed | [`lr-transfer-250M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/lr-transfer-250M) |
| [8k-lr-sweep-60M](8k-lr-sweep-60M.html) | 15 | 60M | LR × seed at 8k context | [`lr-sweep-8k-60M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/lr-sweep-8k-60M) |
| [moe-lr-sweep-8k](moe-lr-sweep-8k.html) | 18 | 60M/125M | LR × seed, top-2 of 8 experts | [`moe-lr-sweep-8k`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/moe-lr-sweep-8k) |
| [batch-size-grid-8k](batch-size-grid-8k.html) | 42 | 60M/125M | batch × LR × seed at 8k, dense and routed | [`batch-size-grid-8k`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/batch-size-grid-8k) |
| [seed-variance](seed-variance.html) | 63 | 60M/125M | seed at a fixed MoE recipe | [`seed-variance-60M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/seed-variance-60M), [`seed-variance-125M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/seed-variance-125M) |
| [duration-ablation](duration-ablation.md) | 42 | 60M/125M | fixed-TPP reference vs cross-horizon duration scaling | [`duration-ablation-60M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/duration-ablation-60M), [`duration-ablation-125M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/duration-ablation-125M) |
| [moe-ablations](moe-ablations.html) | 23 | 60M/125M/250M | learned biases; router auxiliary-loss coefficient | [`moe-no-bias`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/moe-no-bias), [`moe-router-aux-125M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/moe-router-aux-125M) |
| [expert-load-scaling](expert-load-scaling.html) | 5 | 125M | per-expert gradient/update scaling by current load | [`moe-expert-load-scaling-125M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/moe-expert-load-scaling-125M) |
| [moe-weight-decay](moe-weight-decay.html) | 36 | 60M/125M | base AdamW weight decay × seed | [`moe-weight-decay`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/moe-weight-decay) |
| [gumbel-local-moe](gumbel-local-moe.html) | 8 | 125M | Gumbel-routed local MoE steps × seed | [`moe-gumbel-local-125M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/moe-gumbel-local-125M) |
| [sparse-autoencoder-eqflop](sparse-autoencoder-eqflop.html) | 13 | 60M geometry | dictionary width × retained width at equal algorithmic FLOPs | [`sparse-autoencoder-eqflop-60M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/sparse-autoencoder-eqflop-60M) |
| [fuzzy-topk-three-arm-ladder](fuzzy-topk-three-arm-ladder.html) | 24 | 60M/125M/250M | dense vs fuzzy TopK vs double-fuzzy at matched active FLOPs | [`fuzzy-topk-three-arm-ladder`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/fuzzy-topk-three-arm-ladder) |
| [fuzzy-topk-sparsity-diagnostics](fuzzy-topk-sparsity-diagnostics.html) | 12 | 60M/125M/250M/500M | per-feature fuzzy TopK activity over training | [`fuzzy-topk-sparsity-diagnostics-ladder`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/fuzzy-topk-sparsity-diagnostics-ladder) |
| [fuzzy-topk-dead-latent-rejected-paths](fuzzy-topk-dead-latent-rejected-paths.html) | 42 | 60M/125M/250M | balance/homeostasis and zero-forward ghost-AuxK | [`fuzzy-topk-balance-homeostasis-rejected`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/fuzzy-topk-balance-homeostasis-rejected), [`fuzzy-topk-ghost-auxk-rejected`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/fuzzy-topk-ghost-auxk-rejected) |
| [fuzzy-topk-reconstruction-125m](fuzzy-topk-reconstruction-125m.html) | 23 | 125M | reconstruction coefficient, literal AuxK, held-out-seed 20-TPP pair | [`fuzzy-topk-reconstruction-125M`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/fuzzy-topk-reconstruction-125M) |
| [transfer-charts](transfer-charts.html) | — | — | derived figures, not a run dashboard | — |

Each study also carries a compact `snapshot.json.gz` (0.05–1.1 MB of thinned
curves) and an explicit expanded `full.json.gz`. The latter is lossless for the
older studies and deliberately bounded only for the reconstruction study.
Some studies carry a separate diagnostic snapshot. Compact snapshots are what
the browser and derived visualizations load before anything larger.

---

## batch-size-sweep-60M.html

75 runs: **5 batches × 5 learning rates × 3 seeds** at 60M, 5 tokens per
parameter, 1,024 context. The widest grid here, and what study 2 leans on.

```bash
for bs in 32 64 128 256 512; do
  for lr in 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625; do
    for seed in 1337 1338 1339; do
      rig run reference --context 1k --cluster v4-32 --profile dev \
        --tier 60m --tokens-per-parameter 5 \
        --batch-size "$bs" --base-learning-rate "$lr" --seed "$seed" \
        --name "60m-bs${bs}-lr${lr}-s${seed}"
    done
  done
done
rig report --runs <batch-sweep-60M> --max-points 1440 --layer-snapshots 400 \
  --output docs/reports/batch-size-sweep-60M.html
```

## lr-batch-sweep-125M.html

27 runs: **3 batches (64/128/256) × 3 learning rates (2^-7/2^-8/2^-9) × 3
seeds** at 125M, 5 TPP, 1,024 context.

The grid is a batch × LR product, so either axis can be read as the subject.
This replaces the former `batch-size-sweep-125M.html` and `lr-sweep-125M.html`,
which were two renderings of these same 27 runs.

```bash
for bs in 64 128 256; do
  for lr in 0.0078125 0.00390625 0.001953125; do
    for seed in 1337 1338 1339; do
      rig run reference --context 1k --cluster v4-32 --profile dev \
        --tier 125m --tokens-per-parameter 5 \
        --batch-size "$bs" --base-learning-rate "$lr" --seed "$seed" \
        --name "125m-bs${bs}-lr${lr}-s${seed}"
    done
  done
done
```

## batch-size-sweep-250M.html

36 runs: **4 batches (64/128/256/512) × 3 learning rates × 3 seeds** at 250M,
5 TPP, 1,024 context.

Three runs — `250m-5tpp-bs512-lr2e-7`, all three seeds — recorded diagnostics
only from step 1920 onward. A report refuses a diagnostics log that does not
start at step 1, because its axes would not line up with the training curve, so
those three carry their partial series as `diagnostics-partial.riglog`: kept
beside the run, not declared, read by nothing automatically. The runs still
plot from their training curves rather than being dropped over it.

```bash
for bs in 64 128 256 512; do
  for lr in 0.0078125 0.00390625 0.001953125; do
    for seed in 1337 1338 1339; do
      rig run reference --context 1k --cluster v4-32 --profile dev \
        --tier 250m --tokens-per-parameter 5 \
        --batch-size "$bs" --base-learning-rate "$lr" --seed "$seed" \
        --name "250m-bs${bs}-lr${lr}-s${seed}"
    done
  done
done
```

## batch-size-sweep-500M.html

12 runs at two token budgets. Run names carry the budget
(`500m-5tpp-…` against `500m-20tpp-…`) because the two are different
experiments whose losses are not comparable to each other.

This is study 3's dashboard. It replaces both the former `500M-20tpp-v6e.html`
(three of these twelve) and `500M-20tpp-diagnostics.html`, which existed only
because those three were once the only 500M runs whose diagnostics could be
read. All twelve can now.

```bash
# 5 TPP arm, batch bracket at the optimal LR
for bs in 128 256; do
  for seed in 1337 1338 1339; do
    rig run reference --context 1k --cluster v4-32 --profile dev \
      --tier 500m --tokens-per-parameter 5 \
      --batch-size "$bs" --base-learning-rate 0.00390625 --seed "$seed" \
      --name "500m-5tpp-bs${bs}-s${seed}"
  done
done

# 20 TPP arm on the v6e-8: batch bracket, then the LR bracket at batch 128
for bs in 64 128 256; do
  rig run reference --context 1k --cluster v6e-8 --profile dev \
    --tier 500m --tokens-per-parameter 20 --checkpoint-policy none \
    --batch-size "$bs" --base-learning-rate 0.00390625 --seed 1337 \
    --name "500m-20tpp-bs${bs}-s1337"
done
for lr in 0.0078125 0.001953125; do
  rig run reference --context 1k --cluster v6e-8 --profile dev \
    --tier 500m --tokens-per-parameter 20 --checkpoint-policy none \
    --batch-size 128 --base-learning-rate "$lr" --seed 1337 \
    --name "500m-20tpp-bs128-lr${lr}-s1337"
done
```

## 3-seed-gradient-spike.html

12 runs: **4 learning rates × 3 seeds** at 250M, batch 128, 5 TPP. Built to
settle the 250M reseed in study 1, and the evidence base for
[GRADIENT_SPIKES.md](../GRADIENT_SPIKES.md).

Its diagnostics were unreadable long-form CSV until they were converted, so
for a while the dashboard about gradient spikes contained no gradient
statistics at all.

```bash
for lr in 0.015625 0.0078125 0.00390625 0.001953125; do
  for seed in 1337 1338 1339; do
    rig run reference --context 1k --cluster v4-32 --profile dev \
      --tier 250m --tokens-per-parameter 5 \
      --batch-size 128 --base-learning-rate "$lr" --seed "$seed" \
      --name "250m-lr${lr}-s${seed}"
  done
done
```

## 8k-lr-sweep-60M.html

15 runs: **5 learning rates × 3 seeds** of
[`reference --context 8k`](../../recipes/reference/) — 60M at 8,192 context with
document masking, batch 16 so tokens per step and step count match the
1,024-context ladder exactly. This is study 4.

```bash
for lr in 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625; do
  for seed in 1337 1338 1339; do
    rig run reference --context 8k --cluster v4-32 --profile dev \
      --tier 60m --tokens-per-parameter 5 \
      --base-learning-rate "$lr" --seed "$seed" --checkpoint-policy none \
      --name "60m-bs16-lr${lr}-s${seed}"
  done
done
```

## Historical MoE optimizer note

Every archived `reference_moe` run in `moe-lr-sweep-8k`, every routed arm in
`batch-size-grid-8k`, and both seed-variance cohorts predate commit
`102a264672c8453700a02e321495a14c585e58ea`. The old AdamW mask inferred decay
from array rank, so stacked rank-2 `expert_up_b` and `expert_down_b` bias
tensors incorrectly received weight decay. We expect the numerical difference
to be minor, but the corrected recipe cannot reproduce those runs bit-for-bit.
The archived metrics remain observations of the pre-fix recipe; the commands
below reproduce the study design with the corrected policy.

## moe-lr-sweep-8k.html

18 runs of [`reference_moe`](../../recipes/reference_moe/) — top-2 of 8
experts at 8,192 context, forked from the dense 8k ladder. 60M at five learning
rates × three seeds, plus 125M spot runs at three learning rates.

The routed ladder peaks at `2^-8`, the same learning rate the dense one does,
and beats it at every learning rate by 0.07–0.12 nats at equal *active*
parameters and matched compute, for about 1.7x the memory. No expert in any of
the 12 layers finished below 1% of assignments in any of the 18 runs.

This report carries six routing series the dense reports do not have: balance
loss, busiest and idlest expert share, routing entropy, mean top-1 gate, and
router logit RMS. They are recorded model-wide and per layer, with per-expert
load for all 8 experts in all 12 layers, at every step.

```bash
for lr in 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625; do
  for seed in 1337 1338 1339; do
    rig run reference_moe --context 8k --cluster v4-32 --profile dev \
      --tier 60m --tokens-per-parameter 5 \
      --base-learning-rate "$lr" --seed "$seed" --checkpoint-policy none \
      --name "60m-moe-lr${lr}-s${seed}"
  done
done
```

## batch-size-grid-8k.html

42 runs extending the two 8k ladders to batch 32 and 64 — `reference --context 8k` and
`reference_moe` at 60M with three seeds per cell, `reference_moe` at 125M with
one. Three learning rates at every batch, so no batch is judged at a rate
picked for another. The batch-16 arm is not in this study: it is the ladder
each family already had, in `lr-sweep-8k-60M` and `moe-lr-sweep-8k`.

The token budget is held fixed across batches, so doubling the batch halves the
optimizer steps — 2,286 down to 571 at 60M. **Batch 16 wins everywhere.** The
best batch-32 run costs 0.39 nats at 60M dense, 0.27 routed, 0.05 at 125M; the
best batch-64 run costs 1.30, 1.15, and 0.31. Throughput is flat across the
grid (1,041 → 1,100 → 1,093 TFLOP/s at 60M dense), so nothing is bought back in
wall-clock. This reverses the 1,024-context ladder, where batch 128 was optimal
and larger batches finished sooner on the same budget; at 8k a single sequence
is eight times longer, so batch 16 already saturates the chips.

The apparent best learning rate moves between cells, but the seed spread grows
with batch — median 0.011 at batch 16, 0.046 at 32, 0.068 at 64 — until it is
as large as the gaps between rates. The drift is not resolvable at three seeds,
and every large-batch cell is far worse than batch 16 at every rate tried, so
it was not worth more machine time.

```bash
for recipe in reference reference_moe; do
  tag=$([ "$recipe" = reference ] && echo 8k || echo moe)
  for batch in 32 64; do
    for lr in 0.0078125 0.00390625 0.001953125; do
      for seed in 1337 1338 1339; do
        rig run "$recipe" --context 8k --cluster v4-32 --profile dev \
          --tier 60m --tokens-per-parameter 5 --batch-size "$batch" \
          --base-learning-rate "$lr" --seed "$seed" --checkpoint-policy none \
          --name "60m-${tag}-bs${batch}-lr${lr}-s${seed}"
      done
    done
  done
done
```

## seed-variance.html

Two incomplete but already substantial fixed-recipe cohorts: **41 of 64**
planned seeds at 60M and **22 of 64** at 125M. The 60M seed-1369 artifact is
excluded because it came from a dirty, different `train.py`; all 63 retained
runs share the same recipe and configuration hashes. Final FineWeb validation
loss is 3.9357 ± 0.0167 at 60M and 3.5814 ± 0.0055 at 125M (mean ± sample SD).

The report defaults to training loss against cumulative training FLOPs. Its top
row plots the mean with a shaded ±1 sample-SD band; its bottom row plots the SD
directly. The selector can switch among all retained training and diagnostic
metrics. Runs were aligned by exact optimizer step and rejected if their column
layout, token accounting, or FLOP accounting differed. Curves longer than
1,440 points were jointly thinned against the mean and SD only after computing
both across-seed statistics. The checked-in HTML is self-contained; no bespoke
plotting script is retained.

Because the seed controls both initialization and shuffled data order, raw
training-loss and gradient SD includes current-batch composition. Fixed-set
final validation is the cleaner endpoint model-variance estimate. The two
cohorts also ran on different TPU topologies, so the paired display is not a
hardware-controlled test of variance scaling with model size.

The two Hugging Face study cards hold the current reproduction commands and
the pre-`102a264672c8453700a02e321495a14c585e58ea` AdamW compatibility note.

## duration-ablation.md

42 runs: two matching 21-run cohorts at 60M and 125M, all at 20 TPP. Each tier
contains a three-point LR bracket for both the fixed-TPP reference and the
cross-horizon duration treatment, plus the treatment's batch-512 iso-horizon
point; every cell has seeds 1337–1339.

The reference keeps its `2^-8` base-LR optimum. The duration rule predicts that
`2^-7` should compensate for its additional fourfold `m_D`, but that point is
worse at both tiers and separated at 125M. Batch 512 is worse at 60M and tied
with duration batch 128 at 125M, where both trail reference. The
[report](duration-ablation.md) gives the full mean ± SD and Welch-comparison
tables, explains why this is not a universal refutation of Complete(d)P, and
links the two full-resolution studies.

The earlier 60M study was not lost: [`batch-size-sweep-60M`](batch-size-sweep-60M.html)
is the separate 75-run batch × LR grid at 5 TPP. The new 60M cohort changes the
horizon to 20 TPP and introduces the duration treatment; it does not duplicate
that grid.

```bash
for tier in 60m 125m; do
  for seed in 1337 1338 1339; do
    for lr in 0.0078125 0.00390625 0.001953125; do
      rig run reference --context 1k --cluster v4-32 --profile dev \
        --tier "$tier" --tokens-per-parameter 20 --batch-size 128 \
        --base-learning-rate "$lr" --seed "$seed" \
        --checkpoint-policy none --name "${tier}-r20-bs128-lr${lr}-s${seed}"
      rig run reference_duration --context 1k --cluster v4-32 --profile dev \
        --tier "$tier" --tokens-per-parameter 20 --batch-size 128 \
        --base-learning-rate "$lr" --seed "$seed" \
        --checkpoint-policy none --name "${tier}-d20-bs128-lr${lr}-s${seed}"
    done
    rig run reference_duration --context 1k --cluster v4-32 --profile dev \
      --tier "$tier" --tokens-per-parameter 20 --batch-size 512 \
      --base-learning-rate 0.00390625 --seed "$seed" \
      --checkpoint-policy none --name "${tier}-d20-iso-bs512-lr2e-8-s${seed}"
  done
done
```

## moe-ablations.html

Two focused studies of the routed 8k recipe, combined in a 72 KiB static
[findings report](moe-ablations.html):

- **18 paired bias runs:** `reference_moe` versus `no_bias_moe` at 60M, 125M,
  and 250M, seeds 1337–1339. Removing every learned bias changes mean
  validation by +0.01429, +0.00107, and +0.00508 nats, respectively. No tier
  improves on average; all three 250M pairs favor the reference.
- **5 router-coefficient runs:** coefficient 0, 0.001, 0.01, 0.03, and 0.1 at
  125M, all seed 1350. Zero and 0.001 collapse experts and lose 0.06856 and
  0.01863 nats against 0.01. Coefficients 0.01 and 0.1 differ by only 0.00057
  nats even though 0.1 produces nearly uniform loads across every layer.

The working decision is therefore to keep learned biases and keep coefficient
0.01. The coefficient sweep is single-seed; a future refinement should
replicate only 0.01 against 0.1. The report compares no same-index experts
across runs because routed expert identities are permutation-symmetric.

The no-bias fork and all source mappings are preserved on the research
branches and in the Hugging Face ledgers. To reproduce the paired design:

```bash
git checkout 75126f9cf957f5b77699e1e0f755b81bca16c322
for tier in 60m 125m 250m; do
  for seed in 1337 1338 1339; do
    for recipe in reference_moe no_bias_moe; do
      rig run "$recipe" --context 8k --cluster v4-32 --profile dev \
        --tier "$tier" --tokens-per-parameter 5 --batch-size 16 \
        --base-learning-rate 0.00390625 --seed "$seed" \
        --checkpoint-policy none --name "${tier}-${recipe}-s${seed}"
    done
  done
done
```

For the auxiliary sweep, `router_aux_coefficient` was varied in
`recipes/reference_moe/dev.yaml`; its study card maps each value to the exact
clean commit and config SHA256. After checking out one mapped commit, the run
command is:

```bash
rig run reference_moe --context 8k --cluster v4-32 --profile dev \
  --tier 125m --tokens-per-parameter 5 --batch-size 16 \
  --base-learning-rate 0.00390625 --seed 1350 \
  --checkpoint-policy none --name "125m-router-aux-<coefficient>-s1350"
```

## expert-load-scaling.html

Five matched seed-1350 runs at 125M and 5 TPP compare the unchanged
coefficient-0.01 MoE baseline with a load factor applied either before Adam's
moments or to the normalized update. The factor for an expert is
`1 + c * (sqrt(8 * current_load) - 1)` at strengths 0.5 and 1.

The static [findings report](expert-load-scaling.html) shows the exact
validation endpoints and two permutation-invariant mechanism summaries.
Gradient scaling is nearly canceled by Adam: the busiest/idlest actual-update
ratio stays at 1.00 and `c=0.5` ties baseline within 0.00030 nats. Direct update
scaling survives (ratio 1.19–1.43) but finishes 0.011–0.021 nats worse. This is
one seed, so it is enough to reject the current rule, not to estimate a precise
effect size.

The intervention commit is `fb0bde826f5a68372d9898d3113e1eeb324b5e5b`:

```bash
for mode in gradient update; do
  for strength in 0.5 1.0; do
    rig run expert_load_moe --context 8k --cluster v4-32 --profile dev \
      --tier 125m --tokens-per-parameter 5 --batch-size 16 \
      --base-learning-rate 0.00390625 --seed 1350 \
      --checkpoint-policy none \
      --name "125m-load-${mode}-c${strength}-s1350" -- \
      --expert-load-scaling-mode "$mode" \
      --expert-load-scaling-strength "$strength"
  done
done
```

## moe-weight-decay.html

Thirty-six verified MoE runs sweep the base AdamW weight-decay coefficient at
60M and 125M, with seeds 1337–1339 at every cell. The 125M bracket extends
through 0.8 after 0.3 won the initial boundary.

The static [findings report](moe-weight-decay.html) shows exact endpoints,
three-seed mean ± sample SD, paired differences, and the relevant optimizer
scaling. At 125M, base decay 0.3 wins all three paired seeds and improves on
the 0.1 default by 0.015184 nats. The 0.3–0.5 basin is broad; 0.6 turns upward
and 0.8 returns almost exactly to the no-decay mean.

At 60M, the raw mean favors 0.1, but the 0.3 result is dominated by one run
with a gradient-norm spike of 20.41 at step 57. The evidence supports 0.3 as a
125M-specific working choice, not yet as a cross-tier default or as proof of
non-transfer. More 60M seeds around 0.1–0.3 would settle that distinction.

The complete grid can be reproduced from commit
`53a15a25f7b20d2701de9730fe17b1a407fdeb16`:

```bash
for tier in 60m 125m; do
  decays="0 0.03 0.1 0.3"
  [ "$tier" = 125m ] && decays="$decays 0.4 0.5 0.6 0.8"
  for seed in 1337 1338 1339; do
    for wd in $decays; do
      rig run weight_decay_moe --context 8k --cluster v4-32 --profile dev \
        --tier "$tier" --tokens-per-parameter 5 --batch-size 16 \
        --base-learning-rate 0.00390625 --seed "$seed" \
        --checkpoint-policy none --name "${tier}-wd${wd}-s${seed}" -- \
        --weight-decay "$wd"
    done
  done
done
```

## gumbel-local-moe.html

Eight verified 125M runs test `K={0,1,2,4}` extra optimization steps inside
every routed block. K=0 and K=2 have matched seeds 1350, 1369, and 1388; K=1
and K=4 are seed-1350 shape probes. Each inner step draws a fresh hard Gumbel
top-2 route, retains clean mixture weights, and applies raw stateless SGD to a
single activation/output-gradient-normalized objective. The ordinary outer
AdamW update still occurs exactly once.

The prose-led [findings report](gumbel-local-moe.html) explains why the primary
result is endpoint-neutral but mechanistically informative. K=2 changes the
three-seed validation mean by +0.000021 nats while costing 1.344× traced FLOPs
and 1.971× training time. Router balance improves modestly, but clean entropy
falls and logit RMS rises, consistent with the router building margins against
the perturbation. More decisively, the first identical actual-update L2 norm
changes by only 4.9 parts per million for K=2: the raw-SGD local vector is tiny
beside the AdamW update it is meant to supplement.

This rejects the exact normalization/SGD realization, not all extra-compute
MoE exploration. The cost multipliers describe this 125M implementation; they
are not a substitute for an algorithmically equi-FLOP comparison. A useful
follow-up should normalize the local delta against the observed outer MoE delta
or its output displacement, record the two contributions separately, and then
compare appropriately sized larger models at matched total FLOPs. Larger
expert matmuls should also move the local path toward compute-bound execution.
Parameter/gradient/update diagnostics in this study are block-scoped only;
per-expert router loads remain complete.

The full grid is clean at commit
`28df45b92ad4f56b0519ea01b4c3f3e95d3b73fa`:

```bash
for cell in \
  0:1350 0:1369 0:1388 \
  2:1350 2:1369 2:1388 \
  1:1350 4:1350
do
  k=${cell%%:*}
  seed=${cell##*:}
  uv run --frozen --no-sync rig run gumbel_local_moe \
    --context 8k --cluster v4-32 --profile dev \
    --tier 125m --tokens-per-parameter 5 --batch-size 16 \
    --base-learning-rate 0.00390625 --seed "$seed" \
    --checkpoint-policy none --name "125m-k${k}-s${seed}" -- \
    --local-moe-steps "$k"
done
```

## sparse-autoencoder-eqflop.html

Thirteen verified seed-1350 runs compare the 60M dense GELU reference with a
3 × 4 grid of overcomplete TopK-ReLU MLPs at 8k context. Stored dictionary
width is `H/D ∈ {4,8,16}` and retained width is
`K/D ∈ {4,2,1,1/4}`. Every point targets the same 221.03-PFLOP full-model
algorithmic training budget.

The sparse MLP charges `2DH` for scoring all dictionary features, `2KD` for
the selected decoder, and `8KD` for its four active-width backward
contractions, for a training total of `2DH + 10KD` per token and layer. The
ordinary 4× dense MLP costs `48D²`. Costs outside this contraction contract—
selection, gathers/scatters, padding, memory traffic, and dense AdamW
state—remain real but do not define the algorithmic comparison.

For each `(H,K)`, one- and two-layer traces identify the affine full-model
relation `F(L)=A+BL`. We choose the nearest integer layer count to the dense
per-token cost, then choose whole schedule steps as
`round(C_dense / (batch × sequence × F(L)))`. Layers range from 11 to 14 and
steps from 2,218 to 2,354; the largest total-compute mismatch is below 0.02%.
Parameter count and TPP vary intentionally, so these are explicit-step
equal-FLOP runs rather than a fixed-TPP duration ladder.

The static [findings report](sparse-autoencoder-eqflop.html) shows the full
derivation and exact grid. `H=16D,K=D,L=12` is best at 3.910170 validation
loss, 0.090275 nats below the 4.000445 dense anchor. Moderate activity is
consistently strongest at `H=16D`; `K=D/4` gives much of the gain back. The
`H=4D,K=4D` dense-ReLU control is 0.016506 nats worse than GELU, so activation
choice alone does not explain the result.

This is a one-seed algorithmic result, not a practical kernel result. The
gathered implementation is roughly 5–57× slower than dense across the grid,
and a Pallas TensorCore prototype was slower still on v4. The study motivates
a better sparse primitive and larger-model replication; it does not claim
current wall-clock efficiency.

The runs are clean at commit
`ad11241342321aee8984e3aa362ffd7b32278993`. A representative winning cell is:

```bash
git checkout ad11241342321aee8984e3aa362ffd7b32278993
uv run --frozen --no-sync rig run sparse_autoencoder \
  --cluster v4-32 --profile dev --tier 60m --context 8k \
  --batch-size 16 --base-learning-rate 0.00390625 --seed 1350 \
  --checkpoint-policy none --name 60m-eqflop-h16d-k1d-l12-s1350 -- \
  --sparse-layers 12 --sparse-mlp-mult 16 --sparse-top-k 384 \
  --sparse-training-steps 2319 --sparse-mlp-backend reference
```

## fuzzy-topk-three-arm-ladder.html

Twenty-four verified paired-seed runs compare dense GELU, an outer fixed-group
fuzzy TopK, and a doubly-fuzzy input-plus-hidden selector at 8k context. The
three-arm result is complete at 60M and 125M; dense and fuzzy also complete at
250M. Every schedule is matched to its dense tier's total active matrix-FLOP
budget within 0.02%, while stored parameters and backend-issued FLOPs are
reported rather than equated.

The static [findings report](fuzzy-topk-three-arm-ladder.html) shows every
endpoint and paired difference. Fuzzy beats dense in all nine paired seeds,
with mean gains of 0.078033, 0.069399, and 0.076030 nats at 60M, 125M, and
250M. Double-fuzzy loses to fuzzy in all six completed pairs by 0.076437 and
0.062993 nats at 60M and 125M, returning approximately to dense quality
despite its larger stored parameter count.

The added signed one-of-four selector sees only the normalized input to the
MLP branch. The full pre-norm residual remains the identity bypass, so the
negative result is about restricting the learned update's input rather than
deleting the model's residual stream. The current masked-choicewise backend
also does not physically skip zero-input UP products; active and issued FLOPs
remain separate quantities.

The first double-fuzzy 250M run exceeded the 3,600-second development-harness
timeout at step 5,970/9,296 and has no result or canonical validation. The
remaining two seeds were not launched. That partial curve, along with every
short timing gate, is excluded from the 24-run archive and all quality means.
There is therefore no three-arm 250M claim.

The exact run commits are `fe2bd0f28888103fb473d0d8f70cca144a5488e9`
(dense), `5d014ccdef84e179f9dc015e0e5f05871800fceb` (fuzzy), and
`fd1d2542bc9d2735902a963a4eb24233526ac4fc` (double-fuzzy). The canonical
[three-arm manifest](../../recipes/fuzzy_topk_autoencoder/ablation-three-arm-ladder-60m-125m-250m-3seed.json)
contains every coordinate, seed, schedule, and run ID. A representative run is:

```bash
git checkout 5d014ccdef84e179f9dc015e0e5f05871800fceb
uv run --frozen --no-sync rig run fuzzy_topk_autoencoder \
  --cluster v4-32 --profile dev --tier 60m --context 8k \
  --batch-size 16 --base-learning-rate 0.00390625 --seed 1337 \
  --checkpoint-policy none \
  --name 60m-eqflop-fuzzy-h16d-k4d-l11-s1337 -- \
  --sparse-top-k 1536
```

## fuzzy-topk-sparsity-diagnostics.html

Twelve verified fuzzy-only runs add full per-feature observations to the 60M,
125M, 250M, and 500M active-compute ladder. Every block records winner
frequency, positive activation frequency, unconditional activation mean, and
unconditional activation RMS for every one of its `H=16D` stored features on
one 131,072-token global batch at step 1, every ten steps, and final.

The observer is trajectory-neutral. All nine 60M–250M `training.riglog` files
are byte-identical to the same-tier, same-seed uninstrumented fuzzy runs in the
three-arm study. The twelve raw vector logs total 40.19 GiB. An additional
0.891 GiB companion retains every feature value on a widening time schedule:
all captures through step 200, then 300, 500, 900, 1700, and doubled gaps plus
the exact final capture.

Almost every feature is observed positive by step 1 and all are observed by
step 10, but features become persistently inactive later. The final fraction
with zero positive activation in ten consecutive sampled batches averages
34.33%, 35.76%, 33.87%, and 36.34% over layers and seeds at 60M, 125M, 250M,
and 500M. Block 1 is the bottleneck: its three-seed dead fraction rises from
69.69% to 85.62% across the ladder while normalized within-group winner entropy
falls from 0.424 to 0.103. Later blocks progressively recover activity.

The [static findings report](fuzzy-topk-sparsity-diagnostics.html) summarizes
those reductions. The study browser provides the requested one-off explorer
with tier/seed/layer selectors, a training-step dragger, dead-latent heatmaps,
activation-frequency ridgelines and histograms, and positive-frequency
quantiles. “Dead” remains a sampled-batch definition, not proof that a feature
never activates elsewhere in the corpus; the study observes under-use and does
not test a corrective intervention.

The canonical manifests are
[`ablation-sparsity-diagnostics-ladder-60m-125m-250m-3seed.json`](../../recipes/fuzzy_topk_autoencoder/ablation-sparsity-diagnostics-ladder-60m-125m-250m-3seed.json)
and
[`ablation-sparsity-diagnostics-ladder-500m-3seed.json`](../../recipes/fuzzy_topk_autoencoder/ablation-sparsity-diagnostics-ladder-500m-3seed.json).
The archived records retain the final 500M seed's notebook-dependency-only
`pyproject.toml`/`uv.lock` dirt; trainer, shared tree, config, data, runtime, and
topology identities are unchanged.

## fuzzy-topk-dead-latent-rejected-paths.html

Forty-two verified runs preserve two attempted responses to late feature death
without copying the nine parent controls already present in the sparsity
study.

The 33-run balance/homeostasis path includes 24 one-seed development and gate
runs plus nine full frequency-floor treatments at 60M, 125M, and 250M over
seeds 1337--1339. The selected floor acts only on deficient encoder-bias
entries. It cuts aligned persistent death from 34.42% to 17.68% at 60M and
35.77% to 20.39% at 125M with validation changes below 0.002 nats. At 250M,
the reduction shrinks from 33.87% to 29.20%, dead groups rise from 8.31% to
8.80%, and validation worsens by 0.01133. The intervention balances winners
but does not robustly preserve whole positive groups, so the path is rejected
as a scale-general solution.

The nine-run ghost-AuxK path contains five 120-step systems gates and four
step-900 mechanism screens at 60M seed 1350. Relative to tied initialization
alone, admitted zero-forward ghost coefficients reduce persistent death by
only 0.29--0.57 percentage points, worsen validation by 0.0058--0.0102, and
add 11--16% training time. It is rejected before laddering. The path has no
reconstruction decoder or target and therefore does not test literal SAE
AuxK; the later reconstruction-decoder experiment is separate.

Both archives carry exact ledgers, full-neuron vectors, compact/full browser
payloads, launch scripts, source tarballs, and a complete Git bundle for the
intentionally unmerged research histories. The external parent mapping is
explicit in the balance analysis JSON.

## fuzzy-topk-reconstruction-125m.html

Twenty-three verified archive entries close the train-only reconstruction
line at 125M: three dense controls, three exact vector-logged fuzzy parents,
fifteen short-horizon reconstruction/AuxK endpoints, and a held-out-seed hero
pair. Six controls intentionally repeat exact run IDs from existing studies so
the reconstruction archive and browser comparison are self-contained.

The decoder reconstructs the stopped RMS-normalized MLP input from the same
selected values and feature identities as the deployed fuzzy path.
Reconstruction gradients reach the encoder and train-only decoder, not the
residual input or deployed DOWN matrix. The decoder adds 72.09M optimized
parameters, is unit-row constrained, and is discarded after training.

At the three-seed short horizon, beta=1/4 is the best predeclared positive
coefficient. It changes loss by -0.003738 nats versus fuzzy beta=0 with a 95%
paired-t interval of [-0.024865, +0.017390], while reducing features inactive
at both exact shared step-4600/final captures from 35.77% to 23.18%.
Beta=1 drives aligned inactivity to 5.65% but worsens loss by +0.022288 nats,
with an interval excluding zero. Literal residual AuxK at beta=1 is slightly
worse on both loss and survival and is rejected. The cadence-matched definition
matters: parent vectors were recorded every 10 steps and treatments every 100,
so their trailing-ten-capture windows are not directly comparable.

The seed-1350 long pair records 3.359723 for dense and 3.296025 for the
selected combined arm: -1.896% relative cross-entropy and a 0.93829 perplexity
ratio. Fresh10 macro loss changes from 3.351686 to 3.324764. There is no
20-TPP fuzzy beta=0 control and only one seed, so neither the endpoint nor its
late re-widening isolates a reconstruction effect. Its normalized canonical
gain is comparable to, slightly smaller than, the short combined gain versus
dense.

The archive preserves 4.16 GB of raw feature vectors and totals about 4.60 GB.
Its browser snapshot is 0.37 MB; its explicitly loaded expanded payload is
5.52 MB and keeps at most 2,048 points per run-series plus 32 layer frames.
The final
[`ablation-reconstruction-decoder-125m.json`](../../recipes/fuzzy_topk_reconstruction/ablation-reconstruction-decoder-125m.json)
freezes every endpoint, interval, run ID, and fidelity boundary.

## transfer-charts.html

Not a run dashboard. It is a self-contained derived visualization built from
the compact Hugging Face curve snapshots. The HTML is retained; a bespoke
plotting script is not.

---

## Rebuilding

Download a study from the dataset and point `rig report` at it:

```bash
hf download quintic/rig-logs --repo-type dataset \
  --include 'batch-sweep-60M/*' --local-dir /tmp/rig-logs
rig report --runs /tmp/rig-logs/batch-sweep-60M \
  --max-points 1440 --layer-snapshots 400 \
  --output docs/reports/batch-size-sweep-60M.html
```

`--max-points 0 --layer-snapshots 0` embeds every recorded sample. That is what
the dataset holds; it makes a much larger file than anything committed here.

## Two runs that are not in the dataset

- `20260816T213609.122328Z-…-37299d66` — a 500M run whose `stdout.log` was
  deleted while the process still held the descriptor, so no `result.json` was
  ever written. Its curves survive in the original archive but nothing records
  what it measured, so it cannot be placed on a chart.
- A `studies` directory inside the 60M archive, which is not a run.
