# Double-fuzzy TopK sparse autoencoder

This is an incremental fork of
[`fuzzy_topk_autoencoder`](../fuzzy_topk_autoencoder/). It adds one scientific
operation: before the UP projection, the normalized residual state is split
into fixed groups of four and only the signed maximum in each group is kept.
The existing `H=16D`, grouped `K=4D` TopK-ReLU and feature-specific DOWN rows
are unchanged.

For one token, let `Q=D/4`, `H=16D`, and `K=4D`:

```text
(v, r) = GroupedSignedMax(x, group_size=4)       # Q values + D-indices
z = sum_q v[q] W_up[r[q], :] + b_up             # H scores
(a, i) = GroupedTopK(ReLU(z), groups=K)          # one of four per group
y = sum_j a[j] W_down[i[j], :] + b_down
```

`GroupedSignedMax` ranks by signed value, not magnitude, and has no inner
ReLU. An all-negative input group therefore retains its least-negative member.
The outer selector has exactly the parent recipe's semantics: reduce each
four-feature group, apply ReLU to its winner, and preserve that winner's
original dictionary identity.

Nothing else is added: there is no router, auxiliary loss, activation
renormalization, shared decoder row, rank-slot decoder, or stochastic
per-step permutation. The transformer, residuals, RMSNorm, initialization,
optimizer, data order, attention, validation, and logging contracts come from
the parent recipe.

## Why the groups are fixed

At initialization, residual coordinates and feature triples are exchangeable.
A one-time random permutation followed by contiguous groups therefore has the
same initial distribution as contiguous groups directly. Keeping the layout
fixed avoids permutation traffic and PRNG state and makes a seed reproducible.
It also lets coordinates and features specialize within their competition
groups. Re-randomizing every step would be a distinct stochastic treatment.

## Backends and what they physically save

All three backends implement the same model and share an explicit custom VJP.
Tests compare the output and gradients for `x`, `W_up`, `b_up`, `W_down`, and
`b_down` with a literal dense-zero oracle.

- `reference` gathers the selected rows for both projections. It is the
  numerical oracle and exposes the irregular-memory baseline.
- `choicewise` is the TPU-oriented default and the quality-ladder backend. It
  zero-fills the signed inner winners, then reuses the parent fuzzy recipe's
  proven choicewise MLP unchanged. Its custom reverse rule issues large regular
  UP contractions and the established four-choice DOWN path. It avoids the
  earlier factorized reverse pass's 16 small inner/outer-choice contractions,
  but it does not skip multiply-adds whose logical input is zero.
- `pallas_up` is a rejected experimental backend. A selected-row Pallas primitive
  skips the three unselected UP rows per input group in the forward pass. Its
  reverse path and the DOWN path retain bounded choicewise contractions, so
  there is no serialized loop over the large outer `K`. The kernel processes
  two adjacent four-coordinate groups together: it loads their shared physical
  eight-row TPU tile once and selects both winners for the whole token block.
- `pallas_up_dx` is an experimental lower-arithmetic variant. It additionally
  skips unselected hidden rows when forming the input cotangent, but must visit
  a `K`-long assignment grid. A non-128-wide cotangent is padded for the Pallas
  tile, sliced back to `D`, and billed at its padded width. The v4 timing gate,
  not its nominal FLOP count, decides whether that trade is useful.

This is the precise meaning of “semantic” versus “physical” sparsity here.
Putting zeros into a regular matrix multiplication does not reduce its issued
multiply-adds. Likewise, merely fusing TopK, UP, and DOWN removes intermediate
traffic but saves matrix FLOPs only if the fused kernel does not execute the
unselected products. A future fully selected kernel could approach the active
count below, but it must also aggregate sparse `dW_up` updates and efficiently
materialize the dense optimizer gradient; fusion by itself is insufficient.

## FLOP contract

For `M` tokens, the selected matrix arithmetic is

```text
forward  = 2 M Q H + 2 M K D
backward = 4 M Q K + 4 M K D
total    = 2 M Q H + 6 M K D + 4 M Q K
         = 36 M D^2                         # Q=D/4, H=16D, K=4D
```

Selection comparisons, memory traffic, padding, and dense AdamW state remain
real systems costs but are not matrix FLOPs. The recipe reports this active
whole-model trace separately from backend-issued arithmetic.

The current backend MLP counts are:

| backend | issued training matrix FLOPs per layer |
|---|---:|
| active lower bound | `36 M D²` |
| `pallas_up` | `168 M D²` |
| `pallas_up_dx` | `M(136 D² + 8 D D_pad)`, `D_pad=ceil(D/128)128` |
| `reference` custom VJP | `120 M D²` |
| `choicewise` | `192 M D²` |

The Pallas hybrid physically realizes the 4× UP-forward reduction, but not the
nominal sparse-gradient and decoder savings. On v4 its 56-stage input-group
accumulator measured only 19.60K tokens/s at the 60M gate, versus 966.06K for
the fuzzy baseline, so it is retained only as fused-kernel research and is not
used for quality runs. The composed choicewise fallback deliberately issues
the full regular UP contractions to preserve MXU utilization.

## Role in the unified three-arm ladder

The canonical research contract is the fuzzy recipe's
[`ablation-three-arm-ladder-60m-125m-250m-3seed.json`](../fuzzy_topk_autoencoder/ablation-three-arm-ladder-60m-125m-250m-3seed.json).
It presents dense, fuzzy TopK, and doubly-fuzzy TopK as one paired-seed study.
This directory remains separate because it owns a distinct model and kernel;
its versioned
[`ablation-ladder-60m-125m-250m-3seed.json`](ablation-ladder-60m-125m-250m-3seed.json)
is the arm-specific launch plan, not a separate research result. It reuses the
already verified dense controls and preserves `H=16D`, `Q=D/4`, and `K=4D` at
every point.

| tier | dense `L,D` | treatment `L,D` | active FLOPs/token | steps | total mismatch | choicewise issued/dense |
|---|---:|---:|---:|---:|---:|---:|
| 60M | `12,384` | `10,448` | 744,326,016 | 2,266 | +0.0190% | 1.421× |
| 125M | `12,640` | `11,704` | 1,376,732,544 | 4,689 | -0.0094% | 1.618× |
| 250M | `16,896` | `14,1024` | 2,709,522,432 | 9,296 | -0.0014% | 1.845× |

Widths are multiples of the inherited 64-wide attention head. The nearby
depths stay close to the preceding fuzzy treatment (`11/11/14`); whole-step
rounding then matches each dense run's total active matrix-FLOP budget.
Parameter count and tokens-per-parameter vary intentionally, so these are
explicit-step active-compute comparisons, not a fixed-TPP ladder.

## Gates and commands

CPU semantics and wiring:

```bash
uv run --frozen --no-sync pytest -q \
  tests/test_double_fuzzy_topk.py \
  tests/test_recipe_double_fuzzy_topk_autoencoder.py
uv run --frozen --no-sync rig run double_fuzzy_topk_autoencoder --profile smoke
JAX_PLATFORMS=cpu .venv/bin/python \
  recipes/double_fuzzy_topk_autoencoder/train.py --profile smoke --print-plan
```

Before the full ladder, run a short v4-32 gate at the smallest treatment
coordinate:

```bash
.venv/bin/rig run double_fuzzy_topk_autoencoder \
  --cluster v4-32 --profile dev --tier 60m --context 8k \
  --batch-size 16 --base-learning-rate 0.00390625 --seed 1350 \
  --stop-after-step 13 --checkpoint-policy none \
  --name 60m-preflight-double-fuzzy-composed-choicewise-s1350 -- \
  --sparse-d-model 448 --sparse-layers 10 --sparse-top-k 1792 \
  --sparse-training-steps 2266 --sparse-mlp-backend choicewise
```

The gate must establish compile success, peak-memory headroom, steady-state
step time, and the absence of a serialized `Q`- or `K`-length fusion tail.
Short-gate loss is not learning evidence. Only after that gate passes should
the manifest's sequential three-seed quality ladder be launched.

The exact sequential ladder command is:

```bash
for cell in \
  60m:448:10:1792:2266 \
  125m:704:11:2816:4689 \
  250m:1024:14:4096:9296
do
  IFS=: read -r tier width layers top_k steps <<<"$cell"
  for seed in 1337 1338 1339
  do
    uv run --frozen --no-sync rig run double_fuzzy_topk_autoencoder \
      --cluster v4-32 --profile dev --tier "$tier" --context 8k \
      --batch-size 16 --base-learning-rate 0.00390625 --seed "$seed" \
      --checkpoint-policy none \
      --name "${tier}-active-eqflop-double-fuzzy-s${seed}" -- \
      --sparse-d-model "$width" --sparse-layers "$layers" \
      --sparse-top-k "$top_k" --sparse-training-steps "$steps" \
      --sparse-mlp-backend choicewise
  done
done
```
