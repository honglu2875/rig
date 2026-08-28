# Fuzzy-TopK sparse-autoencoder prototype

This research fork changes exactly one scientific operation in
[`sparse_autoencoder`](../sparse_autoencoder/): exact global TopK is replaced
by one winner from each fixed feature group. The transformer, feature-specific
decoder rows, parameter initialization, optimizer, data order, attention,
residuals, evaluation contract, and YAML coordinates are otherwise unchanged.
In particular, the inherited stored default remains `H=16D, K=128`; the
performance gate explicitly selects the parent README's `H=16D, K=4D`
coordinate.

For normalized token state `x`, `H=16D`, `K=4D`, and `G=H/K=4`:

```text
z = x W_up + b_up                         # [..., H]
z_group = reshape(z, [..., K, G])
winner = argmax(z_group, axis=G)          # one winner per group
a = ReLU(max(z_group, axis=G))            # [..., K]
i = arange(K) * G + winner                # original feature identities
y = sum_j a[j] W_down[i[j], :] + b_down
```

Thus this remains an encoder dictionary with a distinct decoder row for every
stored feature. It does **not** share decoder rows, assign decoder rows to rank
slots, add a router, renormalize activations, or add an auxiliary objective.
Tests compare the complete operation and gradients for `x`, both weights, and
both biases against a literal dense-hidden construction.

## Why fixed groups are the random reindexing

At random initialization, feature columns of `W_up`, entries of `b_up`, and
the corresponding rows of `W_down` are exchangeable. Randomly permuting those
triples once and then making contiguous groups therefore has exactly the same
initial distribution as making contiguous groups directly. A runtime
permutation would add activation traffic and PRNG state without improving that
distribution.

The grouping is deterministic after initialization, which preserves run
reproducibility. Features can specialize within their fixed competition group;
that is the intended approximation. Re-randomizing groups every step would be
a different, stochastic treatment.

For a random global TopK set, the expected overlap is the number of groups
containing at least one global winner. At the 60M shape (`H=6144`, `K=1536`,
group size four), expected recall is approximately

```text
1 - C(H-K, 4) / C(H, 4) = 68.37%.
```

The approximation loses global winners that collide in one group and admits
the best non-global feature from an empty group. It always returns exactly
`K` slots; all-negative groups contribute zero.

## Two identical execution paths

`run.kernels.sparse_mlp_backend` selects only an implementation, not a model:

- `reference` performs the literal feature-index gather and is the numerical
  oracle. It retains the original irregular decoder behavior.
- `choicewise` is the TPU-oriented default. It reshapes the parameters to
  `[K,G]`, loops over the small static `G=4` axis, masks the winning groups,
  and performs regular `[tokens,K] @ [K,D]` contractions. Its custom reverse
  rule does the same for `dX`, `dW_up`, and `dW_down`. It has no global TopK,
  no loop over `K`, and no `[tokens,K,D]` selected-row tensor.

The choicewise path intentionally trades additional dense MXU work for regular
execution. For `M` tokens its physical MLP training work is

```text
12 M D H = 192 M D^2                  # H = 16D
```

versus `48 M D^2` for the reference 4× dense GELU MLP and the idealized
`2MDH + 10MKD = 72 M D^2` selected-row sparse algorithm. The tracer bills the
actual `192 M D^2`; it does not claim the sparse ideal while executing dense
choicewise contractions.

This is a deliberate systems hypothesis: four regular contraction groups may
finish much sooner than a nominally cheaper graph dominated by thousands of
small gather/scatter fusions. The timing gate below tests that hypothesis;
quality remains unmeasured.

## Matched comparison

The dense control must be invoked at the same explicit coordinates; the dense
recipe otherwise defaults to 1k context while this fork defaults to 8k.

| coordinate | dense control | fuzzy treatment |
|---|---|---|
| recipe | `reference` | `fuzzy_topk_autoencoder` |
| context | 8k, document masked | identical |
| global batch | 16 | identical |
| base LR | `0.00390625` | identical |
| seed | 1350 for the mechanism gate | identical |
| MLP | 4× GELU | `H=16D`, `K=4D`, group size 4 |
| decoder identity | dense coordinate | one row per original fuzzy-selected feature |

Fixed-TPP duration still uses stored parameter count, exactly as in
`sparse_autoencoder`. Since the choicewise backend executes more physical
matrix FLOPs than the earlier ideal sparse accounting, an equal-physical-FLOP
quality study must solve a new step horizon rather than reuse the old 2,286-
step dense or 2,316-step sparse coordinates.

## v4-32 timing gate

On 2026-08-26, one captured steady-state step at the matched 60M, 8k,
batch-16 coordinate measured:

| MLP path | step time | throughput | slowdown vs dense |
|---|---:|---:|---:|
| dense 4× GELU | 98.05 ms | 1.337M tokens/s | 1.00× |
| exact global TopK, `H=16D, K=4D` | 5,816.97 ms | 22.53K tokens/s | 59.33× |
| fuzzy choicewise, `H=16D, K=4D` | 135.68 ms | 966.06K tokens/s | 1.38× |

The approximation is 42.87× faster than the matched exact path. Its traced
physical work is 1.345× the dense full-step work, close to its 1.384× latency
ratio; the regular contractions are therefore operating near the dense
baseline's efficiency. The fuzzy trace contains one bounded four-choice loop
per layer, with no global TopK or selected feature-row gather. Compile times
were 38.07 seconds fuzzy, 36.52 seconds exact, and 31.90 seconds dense.

These are systems measurements, not learning evidence. In particular, loss
from 13 optimizer steps is not a quality result.

## Three-seed quality ablation

The versioned
[`ablation-60m-3seed.json`](ablation-60m-3seed.json) defines the first quality
test. It uses paired seeds 1337–1339 and the original sparse study's
equal-algorithmic-FLOP coordinate:

| coordinate | dense control | fuzzy treatment |
|---|---:|---:|
| layers | 12 | 11 |
| MLP | 4D GELU | 16D dictionary, grouped K=4D |
| steps | 2,286 | 2,316 |
| tokens | 299,630,592 | 303,562,752 |
| semantic matrix FLOPs | 221.030P | 221.066P |
| physically executed matrix FLOPs | 221.030P | 280.152P |

The 0.016% semantic-compute mismatch is inherited from whole-step rounding.
The larger executed count is transparent systems overhead from replacing
indirect decoder operations with regular contractions; shortening the fuzzy
training horizon for that implementation detail would confound the quality
question. Three already completed dense runs are reused only because their
trainer/config hashes and every scientific coordinate exactly match this
manifest. All three verify cleanly.

## 125M and 250M ladder continuation

The versioned
[`ablation-ladder-125m-250m-3seed.json`](ablation-ladder-125m-250m-3seed.json)
extends the paired-seed test to the next two dense ladder geometries. At each
tier, the integer fuzzy depth nearest the dense FLOPs-per-token anchor is
selected first. The fuzzy schedule is then rounded to the nearest complete
global step at the dense run's total semantic matrix-FLOP budget.

| tier | dense L / steps | fuzzy L / steps | dense tokens | fuzzy tokens | semantic mismatch | fuzzy physical / dense |
|---|---:|---:|---:|---:|---:|---:|
| 125M | 12 / 4,709 | 11 / 4,656 | 617,218,048 | 610,271,232 | +0.0088% | 1.390× |
| 250M | 16 / 9,325 | 14 / 9,402 | 1,222,246,400 | 1,232,338,944 | +0.0038% | 1.503× |

Both fuzzy points retain the same `H=16D`, `K=4D`, group-size-four mechanism
as the 60M treatment: `H=10,240`, `K=2,560` at 125M and `H=14,336`,
`K=3,584` at 250M. The larger physical counts again measure the regular
choicewise contractions and do not shorten the scientific learning horizon.

Dense controls for seeds 1337–1339 at both tiers already exist. They are reused
only after artifact verification and exact agreement on trainer/config hashes,
dataset and validation contract, seed, topology, batch, LR, context, and the
ordinary 5-TPP dense schedule. The continuation therefore queues six fuzzy
runs, ordered 125M before 250M and by increasing seed within each tier.

## Unified dense / fuzzy / doubly-fuzzy research

The canonical sealed three-arm contract is
[`ablation-three-arm-ladder-60m-125m-250m-3seed.json`](ablation-three-arm-ladder-60m-125m-250m-3seed.json).
It presents the completed dense and fuzzy runs together with the completed
60M/125M portion of the
[`double_fuzzy_topk_autoencoder`](../double_fuzzy_topk_autoencoder/) arm as one
paired-seed research ladder. The implementation recipes remain separate: this
directory owns fuzzy TopK, while the sibling directory owns doubly-fuzzy TopK.

`Tier` is the dense-equivalent active-compute ladder label, not the stored
parameter count of every arm. `H` is stored MLP width, `Q` is the active input
width into UP, and `K` is the active hidden width into DOWN.

| tier | dense `L` | dense `D` | dense `H` | fuzzy `L` | fuzzy `D` | fuzzy `H` | fuzzy `K` | double `L` | double `D` | double `Q` | double `H` | double `K` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60M | 12 | 384 | 1,536 (`4D`) | 11 | 384 | 6,144 (`16D`) | 1,536 (`4D`) | 10 | 448 | 112 (`D/4`) | 7,168 (`16D`) | 1,792 (`4D`) |
| 125M | 12 | 640 | 2,560 (`4D`) | 11 | 640 | 10,240 (`16D`) | 2,560 (`4D`) | 11 | 704 | 176 (`D/4`) | 11,264 (`16D`) | 2,816 (`4D`) |
| 250M | 16 | 896 | 3,584 (`4D`) | 14 | 896 | 14,336 (`16D`) | 3,584 (`4D`) | 14 | 1,024 | 256 (`D/4`) | 16,384 (`16D`) | 4,096 (`4D`) |

Dense activates all `H=4D`. Fuzzy TopK applies the outer fixed-group
one-of-four selector only. Doubly-fuzzy TopK additionally applies a signed
inner one-of-four selector, so UP sees `Q=D/4`; its outer selector remains
exactly the fuzzy arm's `K=4D` mechanism. Both selectors preserve original
coordinate or feature identity.

| tier | arm | stored parameters | active FLOPs/token | issued FLOPs/token | steps | quality-run status |
|---|---|---:|---:|---:|---:|---|
| 60M | dense | 59,918,208 | 737,673,984 | 737,673,984 | 2,286 | 3/3 complete, verified |
| 60M | fuzzy | 97,123,584 | 728,236,800 | 922,878,720 | 2,316 | 3/3 complete, verified |
| 60M | doubly fuzzy | 117,429,312 | 744,326,016 | 1,057,424,256 | 2,266 | 3/3 complete, verified |
| 125M | dense | 123,456,640 | 1,371,014,400 | 1,371,014,400 | 4,709 | 3/3 complete, verified |
| 125M | fuzzy | 226,753,280 | 1,386,743,040 | 1,927,415,040 | 4,656 | 3/3 complete, verified |
| 125M | doubly fuzzy | 267,270,784 | 1,376,732,544 | 2,227,209,600 | 4,689 | 3/3 complete, verified |
| 250M | dense | 244,444,032 | 2,701,133,568 | 2,701,133,568 | 9,325 | 3/3 complete, verified |
| 250M | fuzzy | 495,053,440 | 2,679,113,472 | 4,027,844,352 | 9,402 | 3/3 complete, verified |
| 250M | doubly fuzzy | 631,835,648 | 2,709,522,432 | 4,999,612,416 | 9,296 | incomplete; no endpoint |

The primary quality comparison is final validation loss at matched **total
active** matrix FLOPs. Issued FLOPs, throughput, elapsed time, and stored
parameters are separate systems and capacity measurements. The explicit step
horizons make each arm's total active-compute mismatch less than 0.02% from
its dense anchor.

The endpoint result is consistent across paired seeds:

| tier | dense mean ± SD | fuzzy mean ± SD | doubly-fuzzy mean ± SD | fuzzy − dense | double − fuzzy |
|---|---:|---:|---:|---:|---:|
| 60M | 4.002994 ± 0.001129 | **3.924961 ± 0.009860** | 4.001398 ± 0.005951 | −0.078033 | +0.076437 |
| 125M | 3.655115 ± 0.012473 | **3.585716 ± 0.005590** | 3.648709 ± 0.006041 | −0.069399 | +0.062993 |
| 250M | 3.384405 ± 0.004746 | **3.308375 ± 0.005207** | — | −0.076030 | — |

Fuzzy TopK beats dense for all nine paired seeds across the three tiers.
Doubly-fuzzy is worse than fuzzy for all six paired seeds at 60M and 125M and
returns approximately to the dense mean despite storing still more parameters.
The inner selector acts only on the normalized input to the MLP branch; the
pre-norm residual bypass remains intact. The evidence therefore says that
discarding three of four coordinates from the learned MLP update erases the
outer fuzzy dictionary's gain, not that the model loses its residual stream.

The first doubly-fuzzy 250M run was interrupted by the development harness's
3,600-second timeout at step 5,970/9,296. It has no `result.json` or canonical
validation endpoint; seeds 1338 and 1339 were not launched after the sequential
queue stopped. The partial curve and every short timing gate are excluded from
the quality result and Hugging Face archive. See the static
[`fuzzy-topk-three-arm-ladder` report](../../docs/reports/fuzzy-topk-three-arm-ladder.html)
for the complete evidence and limitations.

## Per-feature sparsity diagnostic rerun

The versioned
[`ablation-sparsity-diagnostics-ladder-60m-125m-250m-3seed.json`](ablation-sparsity-diagnostics-ladder-60m-125m-250m-3seed.json)
reruns only the successful fuzzy arm at the same 60M, 125M, and 250M
coordinates and paired seeds. Its purpose is mechanism measurement, not a new
architecture comparison. On optimizer step 1, every 10 steps, and the exact
final step, every block records four full `H`-feature vectors over the sampled
global batch of `T=16*8192` tokens:

```text
winner_frequency[f]     = sum_t 1[i_t = f] / T
activation_frequency[f] = sum_t 1[i_t = f and a_t > 0] / T
activation_mean[f]      = sum_t a_t 1[i_t = f] / T
activation_rms[f]       = sqrt(sum_t a_t^2 1[i_t = f] / T)
```

`winner_frequency` separates losing within-group competition from being
selected before ReLU but remaining non-positive. `activation_frequency==0`
identifies a feature that is dead in that sampled batch; accumulating the same
test across captures identifies features never observed active. The activation
moments provide conditional mean and RMS after division by activation
frequency. A positive activation is also the structural data-gradient support
for that feature's `W_up` column and `W_down` row on the sampled batch. This
does not imply that its AdamW parameter delta is literally zero: momentum and
weight decay may still move a parameter with zero current-batch data gradient.

The raw append-only `fuzzy_sparsity.rigvec` tensor is float32 with shape
`[capture, 4, L, H]`. It retains all neuron identities for offline plots. The
study browser embeds only derived `[capture, layer]` summaries—batch-dead and
sampled-never-active fractions, positive-group fraction, within-group winner
entropy and maximum share, activation-frequency quantiles, and conditional
activation moments—so opening the browser never downloads gigabytes of raw
vectors. The browser's raw-log export fetches `.rigvec` files on explicit user
request.

This logging is sampled densely enough to form a useful temporal dataset while
keeping the training cost bounded:

- Non-capture steps invoke the pre-existing training executable and do not
  form or return feature statistics.
- Capture steps first run a stateless forward-only observer, then invoke the
  exact same ordinary/model-diagnostic update executable as every corresponding
  uninstrumented step. The observer repeats one transformer forward at 10% of
  steps, but performs no backward pass or optimizer work. Its feature reductions
  share the four decoder-choice passes inside that observation. A whole-update
  CPU test checks bitwise equality of parameters, Adam state, loss, gradients,
  and ordinary diagnostics with and without first invoking the observer.
- Only the controller transfers the replicated `[4,L,H]` tensor, immediately
  appends it, and retains no device-side history. The growing file uses the
  harness's hidden temporary-file convention, so opportunistic salvage does not
  rescan it every minute; it becomes visible through one atomic rename after a
  successful training loop.

An earlier prototype returned the feature vectors as an auxiliary from the
differentiated update. Its 60M cadence-10 timing was not slower than its control,
but the extra reductions caused XLA to choose a slightly different BF16 update
path and the trajectories diverged after step 20. That implementation and both
short diagnostic runs are explicitly excluded. The separate observer above was
adopted because a logger must not perturb the scientific trajectory; CPU smoke
controls produce byte-identical training logs and bitwise-identical checkpoint
arrays with the observer off and on.

At cadence 10, the expected raw-log volumes are:

| tier | captures/run | transfer/capture | raw/run | three seeds |
|---|---:|---:|---:|---:|
| 60M (`L=11,H=6,144`) | 233 | 1.03 MiB | 240.28 MiB | 720.84 MiB |
| 125M (`L=11,H=10,240`) | 467 | 1.72 MiB | 802.66 MiB | 2.35 GiB |
| 250M (`L=14,H=14,336`) | 942 | 3.06 MiB | 2.82 GiB | 8.45 GiB |

The roughly 11.51 GiB total is intentional: raw neuron identities and dense
temporal coverage are the research product. Before the nine-run queue starts,
v4-32 short runs compare cadence 0 with the production cadence 10 at identical
coordinates. The ladder proceeds if the measured cadence-10 excess is at most
20% of training time and the ordinary optimizer trajectory remains unchanged.

## Commands

CPU wiring and deterministic plan:

```bash
uv run --frozen --no-sync rig run fuzzy_topk_autoencoder --profile smoke
JAX_PLATFORMS=cpu .venv/bin/python \
  recipes/fuzzy_topk_autoencoder/train.py --profile smoke --print-plan
```

Short v4 timing gate:

```bash
.venv/bin/rig profile reference \
  --cluster v4-32 --profile dev --tier 60m --context 8k \
  --batch-size 16 --base-learning-rate 0.00390625 --seed 1350 \
  --stop-after-step 13 --xprof-start-step 11 --xprof-steps 1 \
  --output-dir profiles/fuzzy-topk-dense-control-60m

.venv/bin/rig profile sparse_autoencoder \
  --cluster v4-32 --profile dev --tier 60m --context 8k \
  --batch-size 16 --base-learning-rate 0.00390625 --seed 1350 \
  --stop-after-step 13 --xprof-start-step 11 --xprof-steps 1 \
  --output-dir profiles/fuzzy-topk-exact-control-60m -- \
  --sparse-top-k 1536

.venv/bin/rig profile fuzzy_topk_autoencoder \
  --cluster v4-32 --profile dev --tier 60m --context 8k \
  --batch-size 16 --base-learning-rate 0.00390625 --seed 1350 \
  --stop-after-step 13 --xprof-start-step 11 --xprof-steps 1 \
  --output-dir profiles/fuzzy-topk-choicewise-60m -- \
  --sparse-top-k 1536

.venv/bin/rig profile fuzzy_topk_autoencoder \
  --cluster v4-32 --profile dev --tier 60m --context 8k \
  --batch-size 16 --base-learning-rate 0.00390625 --seed 1350 \
  --stop-after-step 13 --xprof-start-step 11 --xprof-steps 1 \
  --output-dir profiles/fuzzy-topk-reference-60m -- \
  --sparse-top-k 1536 --sparse-mlp-backend reference
```

Do not interpret short-gate loss. The commands above reproduce the historical
systems checks; the sealed quality evidence comes only from the 24 complete,
verified runs named in the canonical manifest.
