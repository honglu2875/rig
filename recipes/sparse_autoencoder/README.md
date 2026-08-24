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

After selection, the TPU decoder in
[`rig/kernels/sparse_mlp.py`](../../rig/kernels/sparse_mlp.py) never constructs
a dense hidden vector full of zeros. It batches tokens into bounded chunks,
prefetches their selected values and indices into SMEM, and uses a Pallas
kernel to DMA only the selected rows of `W_down` for each 128-wide output tile.
Token chunks are vmapped into a parallel kernel-grid dimension. A custom VJP
loops over one selected slot at a time, so dX, dW_up, and dW_down also avoid a
`[tokens, hidden]` cotangent and a `[tokens, k, d_model]` temporary. AdamW still
owns dense parameter and moment arrays; sparsity does not remove that storage.

The small pure-JAX path is the CPU fallback and correctness oracle. Tests
compare Pallas interpreter output and every gradient against the literal
`dense -> TopK-ReLU -> dense` definition. A real-TPU compile and timing gate is
still required because an exact unstructured gather can be bandwidth-bound
even when it performs fewer arithmetic operations.

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
Official and dev select the Pallas sparse decoder; smoke selects the exact
reference backend. The non-smoke tier names preserve the source transformer's
depth/width geometry, not its old dense parameter count:

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

`--sparse-mlp-backend reference` is an implementation-only comparison knob;
it preserves the exact math while replacing the Pallas decoder with JAX
gather/einsum. It is intended for the timing gate, not as a study coordinate.

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

Before this queue, a ten-step trajectory-preserving run must pass real-TPU
compilation, finite-loss checks, and a kernel timing comparison. The full grid
is not launched if the Pallas path is slower than the exact gathered JAX
reference or exhibits a compile/runtime failure; that outcome is a kernel
result, not permission to silently substitute block sparsity.

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
