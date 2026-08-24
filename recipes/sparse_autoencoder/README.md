# Sparse-autoencoder MLP prototype

This research fork keeps the dense [`reference`](../reference/) transformer,
optimizer, data contract, attention, and evaluation code, but replaces every
4× GELU MLP with an overcomplete TopK-ReLU dictionary. It defaults to the `8k`
context preset; `--context 1k` remains available.

For normalized token state `x`, one block computes

```text
z = x W_up + b_up                    # score every stored feature
(a, i) = TopK(ReLU(z), k)            # k values and feature indices per token
y = sum_j a[j] W_down[i[j], :] + b_down
```

The default dictionary is 16× `d_model`, with `k=128`. ReLU precedes TopK.
Consequently, a row with fewer than `k` positive scores may select additional
zeros, but those entries have zero output and zero gradient. Selection indices
are nondifferentiable; gradients flow through the retained positive values.

## What is actually sparse

Exact TopK cannot avoid scoring the complete dictionary: without a separate
router, every `x · W_up[:, h]` is needed to know whether feature `h` belongs to
the top k. The implementation does not hide that cost.

After selection, neither decoder constructs a dense hidden vector full of
zeros. The default gathered-JAX path indexes only the selected rows of
`W_down` and contracts those with the retained values. The Pallas prototype in
[`rig/kernels/sparse_mlp.py`](../../rig/kernels/sparse_mlp.py) instead batches
tokens into bounded chunks, prefetches values and indices into SMEM, pipelines
the physical eight-row BF16 tiles containing the selected rows, and keeps an
FP32 output accumulator in VMEM. A custom VJP shared by both paths loops over
one selected slot at a time, so dX, dW_up, and dW_down avoid both a
`[tokens, hidden]` cotangent and a `[tokens, k, d_model]` temporary. AdamW still
owns dense parameter and moment arrays; sparsity does not remove that storage.

Tests compare Pallas interpreter output and every gradient against the literal
`dense -> TopK-ReLU -> dense` definition. The real-v4 gate found that the
compiler's gathered-JAX path is substantially faster than this TensorCore
Pallas implementation:

| exact decoder | 60M/16x/k128 throughput |
|---|---:|
| gathered JAX | 134.91K tokens/s |
| Pallas, 128-wide tiles | 18.82K tokens/s |
| Pallas, full 384-wide tile | 35.93K tokens/s |

The Pallas runtime on this v4 exposes no SparseCore gather backend, and a
bounded indirect TensorCore DMA is not supported. Official and dev therefore
use gathered JAX. The slower Pallas path remains as a tested prototype and
implementation comparison; it is not used for the mechanism sweep.

The algorithmic MLP training cost per token and layer is

```text
2 D H + 10 K D
```

where `H = mlp_mult × D`: dense dictionary scoring costs `2 D H`; sparse
decoder forward plus dValues, dX, dW_up, and dW_down cost five contractions of
`2 K D`. For comparison, the reference 4× dense MLP costs `48 D²`. The default
16×/k=128 setting is about 0.74× that MLP cost at the 60M geometry and 0.71× at
125M, before hardware padding, selection, gathers, and dense AdamW updates.
The traced FLOP report applies this contract at the named sparse-MLP boundary.

## Configuration and parameter counts

`config.yaml`, `dev.yaml`, and `smoke.yaml` are complete standalone documents.
Official, dev, and smoke select the exact gathered-JAX decoder. The non-smoke
tier names preserve the source transformer's depth/width geometry, not its old
dense parameter count:

| tier geometry | layers | width | heads | stored parameters at 16× |
|---|---:|---:|---:|---:|
| 60M | 12 | 384 | 6 | 102,440,832 |
| 125M | 12 | 640 | 10 | 241,513,600 |
| 250M | 16 | 896 | 14 | 552,897,408 |
| 500M | 19 | 1,280 | 20 | 1,250,004,480 |
| 1B | 21 | 1,792 | 28 | 2,608,872,448 |

Fixed-TPP duration uses this stored parameter count because every encoder
column is scored. A dictionary-width override therefore changes the token
horizon; a TopK override does not. CompleteP width/depth scaling remains
anchored to the source tier geometry and is intentionally not retuned in this
mechanism study.

The two recipe-local research overrides belong after the harness boundary:

```bash
uv run --frozen --no-sync rig run sparse_autoencoder \
  --profile dev --tier 60m --context 8k -- \
  --sparse-mlp-mult 16 --sparse-top-k 128
```

`--sparse-mlp-backend pallas` and `--sparse-mlp-output-block` are
implementation-only comparison knobs. The former selects the slower custom
decoder; the latter changes only its physical output tile. They are intended
for timing gates, not as study coordinates.

## First mechanism grid

The initial screen holds the 60M geometry, 16× dictionary, 8k context, 5 TPP,
global batch 16, base LR `0.00390625`, seed 1350, FineWeb-8B train/validation
contract, and no checkpoint fixed. Only `k` varies:

| `k` | role |
|---:|---|
| 32 | very sparse decoder |
| 64 | lower-active probe |
| 128 | primary default |
| 256 | higher-active probe |

The ten-step trajectory-preserving gate passed finite-loss checks for both
implementations and selected gathered JAX on measured throughput. The sweep
retains exact unit-level TopK semantics; it does not silently substitute block
sparsity to make Pallas faster.

Use the harness rather than invoking `train.py` directly:

```bash
uv run --frozen --no-sync rig run sparse_autoencoder --profile smoke
uv run --frozen --no-sync rig run sparse_autoencoder \
  --cluster v4-32 --profile dev --tier 60m --context 8k \
  --tokens-per-parameter 5 --batch-size 16 --seed 1350 \
  --checkpoint-policy none --name 60m-16x-k128-s1350 -- \
  --sparse-mlp-mult 16 --sparse-top-k 128
```

All ordinary run artifacts and validation-contamination safeguards are
identical to `reference`.
