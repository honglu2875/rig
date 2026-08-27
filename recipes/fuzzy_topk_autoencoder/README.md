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

The canonical three-arm contract is
[`ablation-three-arm-ladder-60m-125m-250m-3seed.json`](ablation-three-arm-ladder-60m-125m-250m-3seed.json).
It presents the completed dense and fuzzy runs together with the pending
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
| 60M | doubly fuzzy | 117,429,312 | 744,326,016 | 1,009,255,296 | 2,266 | pending v4-32 gate |
| 125M | dense | 123,456,640 | 1,371,014,400 | 1,371,014,400 | 4,709 | 3/3 complete, verified |
| 125M | fuzzy | 226,753,280 | 1,386,743,040 | 1,927,415,040 | 4,656 | 3/3 complete, verified |
| 125M | doubly fuzzy | 267,270,784 | 1,376,732,544 | 2,096,366,976 | 4,689 | pending v4-32 gate |
| 250M | dense | 244,444,032 | 2,701,133,568 | 2,701,133,568 | 9,325 | 3/3 complete, verified |
| 250M | fuzzy | 495,053,440 | 2,679,113,472 | 4,027,844,352 | 9,402 | 3/3 complete, verified |
| 250M | doubly fuzzy | 631,835,648 | 2,709,522,432 | 4,647,290,880 | 9,296 | pending v4-32 gate |

The primary quality comparison is final validation loss at matched **total
active** matrix FLOPs. Issued FLOPs, throughput, elapsed time, and stored
parameters are separate systems and capacity measurements. The explicit step
horizons make each arm's total active-compute mismatch less than 0.02% from
its dense anchor.

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

Do not interpret short-gate loss. First establish compile success, steady step
time, memory headroom, and the absence of a `K`-sized serialized fusion tail;
only then solve an equal-physical-FLOP schedule and run a one-seed quality
test.
