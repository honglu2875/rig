# TPU kernels

The shared kernel package contains three trainable building blocks: causal TPU
FlashAttention, exact TopK-ReLU selection, and a memory-bounded tied output
projection with cross entropy. They are common infrastructure; model and
optimization choices remain in each recipe's `train.py`.

Both kernels have dense correctness oracles and explicit static tile sizes.
Treat the defaults as good seeds for the repository's GPT-2-small shape, not as
universal optima.

The attention implementation starts from JAX's Apache-2.0 TPU
[FlashAttention implementation](https://github.com/jax-ml/jax/blob/jax-v0.11.0/jax/experimental/pallas/ops/tpu/flash_attention.py)
and uses a Splash-like static API. The design review also used the
[FlashAttention-4 paper](https://arxiv.org/abs/2603.05451) and its public GPU
implementation as scheduling references. The transferable ideas are tiled IO,
online softmax, independent query/KV work, and recomputation in backward. GPU
warp specialization, TMEM/DSMEM exchange, CTA pairing, and Blackwell-specific
polynomial exponentiation are deliberately not imitated: Pallas exposes TPU
HBM-to-VMEM DMA, VREGs, the VPU, and the MXU through a different execution
model. See JAX's [Pallas TPU
details](https://docs.jax.dev/en/latest/pallas/tpu/details.html) for those
layout and memory constraints.

## TPU FlashAttention

[`rig.kernels.tpu_flash_attention`](../rig/kernels/tpu_flash_attention.py)
uses logical `[batch, heads, sequence, head_dim]` inputs and returns the same
shape. It currently implements dense causal self-attention with matching q/k/v
shapes, `head_dim <= 128`, and a head dimension divisible by 8. Sequence lengths
are transparently padded to a multiple of 128 and sliced back to their logical
length.

The public factory API is:

```python
from rig.kernels import AttentionConfig, make_causal_attention

attention = make_causal_attention(
    AttentionConfig(backend="tpu_flash")
)
output = attention(q, k, v)  # q, k, v are BHSD arrays
```

`AttentionConfig` accepts:

- `backend="tpu_flash"` for the custom Pallas forward, dQ, and dK/dV kernels;
- `backend="jax_flash"` for JAX 0.11's trainable Pallas implementation;
- `backend="reference"` for the dense pure-JAX correctness oracle; and
- `backend="auto"`, which selects `jax_flash` on TPU and `reference` elsewhere.

Select `tpu_flash` explicitly when benchmarking the custom kernel. The
convenience `causal_attention(q, k, v, config=...)` has the same behavior.
`reference_causal_attention` is useful for small numerical tests but
materializes the complete attention matrix.

`AttentionTiles` is the static ten-field tile plan. It separately represents
the forward q/kv major and compute blocks, dK/dV q/kv major and compute blocks,
and dQ q/kv and kv-compute blocks. `attention_tile_candidates` returns a bounded
set of legal TPU candidates; `select_attention_tiles` returns a deterministic
shape heuristic without benchmarking.

## Exact TopK-ReLU selection

[`rig.kernels.sparse_mlp`](../rig/kernels/sparse_mlp.py) supplies the
`sparse_autoencoder` recipe's dense encoder, exact TopK-ReLU support, and
decoder. The production v4 choice is explicit:

```python
from rig.kernels import SparseMlpConfig, sparse_topk_mlp

output = sparse_topk_mlp(
    x, up_weight, up_bias, down_weight, down_bias,
    config=SparseMlpConfig(
        top_k=1536,
        backend="pallas_masked",
        token_block=128,
    ),
)
```

`pallas_masked` requires BF16 activations and a dictionary width divisible by
128. Nonnegative BF16 bit patterns preserve numerical order. The Pallas kernel
packs each value with the reverse column index, reproducing `lax.top_k`'s
stable lower-index tie break, and binary-searches the unique packed kth key in
VMEM. It emits a dense array containing the exact selected values and zero
elsewhere. Its custom VJP masks the activation cotangent, including ReLU's zero
derivative for nominally selected zeros.

The name is intentionally literal: the selector is sparse, but the following
decoder is a dense zero-masked MXU matmul. TPU v4 has no efficient
token-dependent element-sparse matrix-multiply path; gathering different
decoder rows for every token moved far more data than simply multiplying the
dense zero-masked activation. Thus `pallas_masked` preserves the TopK algorithm
and zero gradients outside its support, but it executes decoder FLOPs for the
complete dictionary. Run provenance records this as
`sparse_mlp_decoder_execution=dense_zero_masked_mxu`, while scientific FLOP
accounting remains backend-independent and bills the selected `k` coordinates.

The configured `token_block` is a preferred maximum. Resolution lowers it in
multiples of eight as dictionary width grows, keeping the input, output, and
packed-key windows inside v4's 16 MiB VMEM. `reference` remains the gathered
JAX oracle with a sparse custom VJP. The older `pallas` selected-row decoder is
retained as an explicit research prototype; it is not the default.

At the 8k comparison anchored to the 60M dense model (`D=384`, `H=6144`,
`K=1536`, one sequence per device), value plus all MLP gradients fell from
446.3 ms with the gathered reference path to 2.77 ms with `pallas_masked`; the
4D dense MLP took 0.54 ms.
A complete synthetic four-device step measured 99.3 ms for the 12-layer 4D
dense control and 124.1 ms for the equi-FLOP 11-layer 16D/K4D model: 80.0% of
dense throughput, versus 1.75% for the former gathered implementation on the
earlier v4-32 run. These timings are systems measurements, not loss results,
and the dense versus gathered accumulation order is not bit-equal.

The anchor name is not a stored-parameter claim. The 12-layer dense control
stores 59,918,208 parameters. The compute-matched 11-layer 16D/K4D treatment
stores 97,123,584; counting only selected MLP columns and rows gives 58,144,512
conventionally active parameters per token. It is therefore close to the dense
control in selected-active parameters, but it is not a 60M-parameter model.

Production-style parameter/optimizer donation measured 91.0 ms and 114.4 ms
for the same pair, preserving 79.6% of dense throughput. The masked backend's
hardware cost is effectively independent of `K`: at `K=128`, its full MLP
value-plus-gradient benchmark remained 2.79 ms, while native gathered autodiff
and the sparse custom VJP took 45.4 ms and 54.2 ms. Smaller-K/deeper ladders can
therefore be equi-FLOP scientifically without being equi-time on v4. Treat
`K`-dependent FLOP accounting as an algorithm contract, not a claim that this
chip skips the corresponding decoder or gradient work.

## Tiled tied cross entropy

[`rig.kernels.linear_cross_entropy`](../rig/kernels/linear_cross_entropy.py)
combines the tied output projection, online log-sum-exp, target selection, and
cross entropy without constructing `[batch, sequence, vocabulary]` logits:

```python
from rig.kernels import (
    tiled_tied_cross_entropy,
    tiled_tied_cross_entropy_losses,
)

loss = tiled_tied_cross_entropy(
    hidden,
    embedding,
    targets,
    semantic_vocab_size=50_257,
    vocab_tile_size=2_048,
)
per_token_loss = tiled_tied_cross_entropy_losses(
    hidden,
    embedding,
    targets,
    semantic_vocab_size=50_257,
    vocab_tile_size=2_048,
)
```

`hidden` is `[..., width]`, the tied `embedding` is
`[storage_vocab, width]`, and `targets` matches `hidden.shape[:-1]`. The custom
VJP recomputes one vocabulary tile at a time in backward, so neither direction
creates full logits. Projection operands default to BF16; logits, online
softmax state, gradients, and final loss use FP32.

`semantic_vocab_size` is deliberately separate from aligned storage. For GPT-2
the model may store 50,304 embedding rows while only token IDs `[0, 50_257)`
receive probability mass or output-head gradients. This changes the modeled
probability distribution, so it is an explicit algorithm choice rather than a
kernel-only optimization. The trainer therefore defaults the semantic size to
the full storage vocabulary for both dense and tiled loss. Recipes may
deliberately choose the 50,257-class alternative, but that changes the modeled
distribution and must remain visible in recipe configuration and run
provenance. `vocab_tile_size=2_048` is the measured v4 seed.

A hand-written Pallas version of this loss was tried and removed. It was
correctness-checked but its value-plus-backward microbenchmark ran slower than
the pure-JAX tiled custom VJP above, so at this scale the fusion does not pay
for itself. The code is in git history (removed 2026-08-17) rather than
carried unused; recreating it from this description is the cheaper path if a
larger vocabulary or a different chip changes that balance.

## Tile resolution and autotuning

Attention tile parameters are static XLA/Pallas compilation parameters.
**Tuning never happens inside `jax.jit`, a Pallas kernel, or the compilation of
the real training executable.** A compiled program cannot benchmark alternate
versions of itself.

Ordinary runs should resolve a plan before constructing the attention factory:

```text
exact shipped lookup-table entry
        ↓ miss
deterministic shape heuristic
```

`resolve_attention_tile_plan(key)` implements this read-only policy and reports
`source` as `shipped` or `heuristic`. Both tiers are pure functions of the key,
which is what lets every process in a multi-host job derive identical tile
constants without communicating. There is deliberately no on-disk cache and no
runtime autotuner: a per-host cache file was the one way two processes could
persist different winners for near-equal candidates and then compile divergent
HLO for the same SPMD program. Shipped
entries match the entire runtime and source fingerprint; they are not loose
shape recommendations. The current custom entry is a directly measured seed,
not the result of an exhaustive custom-kernel sweep.

Candidate generation is deliberately bounded rather than Cartesian. It checks
128-wide TPU alignment, padded-sequence divisibility, head shape, major/compute
divisibility, and a conservative live-VMEM model capped at 62.5% of TPU v4's
16 MiB. Compilation or resource failures are retained as failed candidate
measurements. Successful candidates are compared by synchronized median and
median absolute deviation; candidates within the 1%/noise band prefer lower
modeled VMEM and deterministic simpler tiles.

### Explicit synthetic bootstrap

Bootstrapping is an **offline** activity, not part of a run. Measure a new
topology or shape here, then promote the winner into `_SHIPPED_TUNINGS` so
every process derives it from the key. On a multi-host slice remember that a
single host cannot initialize the TPU at all -- the slice spans every VM -- so
measurement has to happen inside a job that owns the whole slice.

```python
import jax.numpy as jnp

from rig.kernels import AttentionConfig, make_causal_attention
from rig.kernels.autotune import (
    autotune_attention,
    make_runtime_key,
)

key = make_runtime_key(
    backend="tpu_flash",
    dtype=jnp.bfloat16,
    batch=8,       # local per-device batch, not global batch
    heads=12,
    sequence=1_024,
    head_dim=64,
    mode="forward_backward",
)

def factory(tiles):
    return make_causal_attention(
        AttentionConfig(backend="tpu_flash", tiles=tiles)
    )

record = autotune_attention(
    key=key,
    attention_factory=factory,
    warmup_runs=2,
    measured_runs=7,
)
```

The bootstrap creates deterministic synthetic q/k/v arrays and an output
cotangent. For every candidate it calls `jax.jit(...).lower(...).compile()`
once, warms the executable, synchronizes every forward-plus-VJP measurement,
and records all samples. It never reads training or validation data, and it
never writes anything: the returned record is yours to inspect.

Microkernel selection is only the first stage. Shortlist the best two or three
candidates, compile a complete representative `train_step` for each, and choose
from synchronized whole-step measurements. Fusion, layout conversion,
collectives, donation, and surrounding computation can erase or reverse a
microbenchmark win. Record this whole-step confirmation alongside the selected
tile plan.

## Key identity and run provenance

A tile plan is only reusable for an exactly matching workload, so the key
covers:

- kernel revision and SHA-256 of the implementation source;
- backend, platform, device kind, global/local device counts;
- JAX, jaxlib, and libtpu versions;
- dtype and exact local `batch × heads × sequence × head_dim` workload;
- forward versus forward-plus-backward mode and causal contract; and
- backward strategy, q/k/v layouts, buffer count, lookahead, exponential mode,
  and conditional-rescale setting.

A shipped entry must match all of it. That is why a plan measured on a v4-8
(`device_count=4`) is correctly ignored on a v4-32 (`device_count=16`) even at
the same model shape: the per-device batch differs.

Every run preserves the key digest, resolved source, winner, and tuning
duration in its result and checkpoint provenance, so the tiles a run compiled
with are always recoverable from the record. Resolution is pure bookkeeping and
happens before `train_seconds`; the timed run still starts from its declared
initial state.

## Measured TPU v4 snapshots

These are per-chip BF16 training microbenchmarks on the repository's locked
JAX 0.11.0, jaxlib 0.11.0, and libtpu 0.0.44.1 runtime. Absolute timings depend
on the benchmark executable, so compare values only within the same group and
rerun after any source or runtime change.

| Group | Exact local workload | Variant | Median |
|---|---|---|---:|
| Attention tile sweep | `B=8, H=12, T=1024, D=64`, forward + backward | dense FP32-score oracle | 3.615 ms |
| Attention tile sweep | same | JAX Flash, all 128 blocks | 6.245 ms |
| Attention tile sweep | same | JAX Flash, `q=512, kv-major=512, compute=256` with corresponding backward blocks | 2.761 ms |
| Direct custom check | same shape, synthetic forward + VJP | TPU Flash, canonical ten-field plan | 2.069 ms |
| Tied-CE tile sweep | `N=8192, D=768, storage V=50304, semantic V=50257`, value + both gradients | dense logits | 14.111 ms |
| Tied-CE tile sweep | same | pure-JAX tiled VJP, vocabulary tile 2,048 | 12.284 ms |
| Explicit Pallas CE experiment | same logical shape, separate benchmark executable | dense / pure-JAX tiled / Pallas tiled | 22.59 / 12.16 / 14.63 ms |

The custom attention result was numerically checked against the dense oracle.
The Pallas CE experiment remains non-production because the simpler pure-JAX
tiled VJP was faster. None of these microbenchmarks substitutes for complete
train-step validation.

An identical full GPT-2-small step (local `B=8` on each of four chips, global
`B=32`, `T=1024`) measured 93.196 ms with dense attention and dense loss,
75.191 ms with custom TPU FlashAttention and dense loss, and 77.048 ms with
custom TPU FlashAttention and tiled loss. These are medians after two warmups
and ten synchronized donated updates. The dense-loss optimized path therefore
reached 435.79k tokens/s in this isolated step benchmark; compilation,
validation probes, diagnostics, and input bookkeeping are not represented by
that number.
