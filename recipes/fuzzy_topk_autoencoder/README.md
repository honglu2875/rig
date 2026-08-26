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
