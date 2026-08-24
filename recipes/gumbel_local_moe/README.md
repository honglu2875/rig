# gumbel_local_moe — local MoE exploration prototype

A research fork of [`reference_moe`](../reference_moe/) that spends additional
optimization work inside each routed block. It keeps the baseline architecture,
data, outer AdamW update, and evaluation contract intact. Passing
`--local-moe-steps 0` selects the exact baseline update path; the prototype
defaults to two local steps.

Like `reference`, this fork has three complete configuration documents:
`config.yaml` for official runs, `dev.yaml` for development runs, and
`smoke.yaml` for the routed CPU check. `--profile` only selects the file.

## Treatment

The outer forward/backward captures each MoE block's residual input, residual
output, and output gradient. For local step `k`, a fresh standard-Gumbel top-k
route explores a different expert assignment while the selected experts retain
their **clean** router-logit mixture weights. There is no sampling temperature.

For layer `l`, the local objective is

```text
0.5 * mean((delta_l / s_l + g_l / q_l) ** 2)
```

where `delta_l` is the sampled block-output displacement from the outer pass,
`g_l` is that block's outer output gradient, `s_l²` is the mean of rolling
input/output second moments, and `q_l²` is the rolling output-gradient second
moment. The moments use the outer optimizer's effective Adam `beta2`. Expanding
the square gives descent through its cross term and an activation-scaled
quadratic restraint through `delta_l²`; one normalization therefore serves
both roles.

The local updates are stateless SGD with the current global learning rate and
the same CompleteP tensor multipliers. They do not create another Adam state,
apply another weight-decay step, or update non-MoE parameters. The ordinary
AdamW update still happens exactly once per batch. The only treatment knob is
the integer local-step count.

Training logs add four per-block series when the treatment is enabled:
`moe.local_loss`, `moe.input_rms`, `moe.output_rms`, and
`moe.output_gradient_rms`. The recorded local loss is the objective immediately
before the final local update in that global step.

## First study grid and result

The first mechanism study is fixed before its full runs. Every arm uses the
125M tier, 8k context, 5 TPP, global batch 16, base LR `0.00390625`, the
FineWeb-8B train/validation contract, the v4-32 cluster, and no checkpoint.
`K=2` is the primary treatment; `K=1` and `K=4` only test step-count shape.

| local steps `K` | seeds | role |
|---:|---|---|
| 0 | 1350, 1369, 1388 | exact `reference_moe` update baseline |
| 1 | 1350 | lower-work mechanism probe |
| 2 | 1350, 1369, 1388 | primary treatment |
| 4 | 1350 | higher-work mechanism probe |

All arms run the complete 4,709-step schedule. The grid is not expanded based
on intermediate losses; further work requires a new declared study.

The grid is complete. Across the three paired K=0/K=2 seeds, final validation
changes by **+0.000021 nats** while K=2 costs **1.344× traced FLOPs** and
**1.971× training time**. K=1 and K=4 are also worse than K=0 at their matching
seed. Clean router entropy falls and logit RMS rises under K=2, while the first
identical actual-update L2 norm differs from K=0 by only 4.9 parts per million.

The working decision is to preserve this as a negative mechanism result and
not carry the current raw-SGD update forward unchanged. It rejects this
normalization, not extra-compute MoE exploration: a follow-up should normalize
the local delta against the observed outer MoE delta or its predicted output
displacement, then test appropriately resized models at matched total FLOPs.
The present wall multiplier is an implementation measurement at 125M, not the
algorithmic criterion; larger expert matmuls should make the local work more
compute-bound. See the prose-led
[`gumbel-local-moe` report](../../docs/reports/gumbel-local-moe.html) and the
full-resolution
[`moe-gumbel-local-125M` archive](https://huggingface.co/datasets/quintic/rig-logs/tree/main/moe-gumbel-local-125M).

| | `reference` | this fork's MoE baseline |
|---|---|---|
| context presets | `1k` default, `8k` optional | same presets, `8k` default |
| MLP | dense 4× GELU | top-2 of 8 experts, 2× each |
| active MLP width/token | 4× | 4× |
| router auxiliary loss | none | 0.01 load-balancing coefficient |
| token batch and schedule | 131,072 tokens/step | identical |

The declared tier parameter count is the **active/equi-FLOP ladder anchor**, not
the total stored MoE parameter count. Top-2 × 2-wide experts matches the dense
4× active MLP computation while eight replicated experts add inactive capacity.

At the native 8k default, batch 16 is the recipe-local reference and therefore
`m_B = 1`. Selecting `1k` selects batch 128 as its anchor, exactly as it does in
the dense recipe. This reanchoring is an explicit project choice documented in
[`docs/COMPLETEP.md`](../../docs/COMPLETEP.md).

AdamW decay follows parameter roles: embeddings and `*_w` tensors decay;
router/expert biases and normalization scales do not. In particular, stacked
expert biases are rank-2 arrays but remain biases—array rank is not a decay
policy.

> **Historical-run compatibility.** The archived MoE studies predate commit
> `102a264672c8453700a02e321495a14c585e58ea`. Before that commit, the AdamW
> mask inferred decay from array rank, so the stacked rank-2 `expert_up_b` and
> `expert_down_b` bias tensors incorrectly received weight decay. We expect the
> numerical effect to be minor, but the corrected recipe cannot reproduce those
> runs bit-for-bit. Treat the archived metrics as observations of the pre-fix
> recipe; reproduction commands use the corrected policy.

## Why masking follows context

A window is cut live from a flat token stream, so it can span several
documents. Measured on a 100M-token FineWeb shard:

| | 1,024 | 8,192 |
|---|--:|--:|
| documents a random window spans | ~1.5 | **~11.8** |
| tokens in documents at least that long | 53.0% | **8.5%** |

At 1k the cross-document surface is about one boundary per window and both
recipes leave it unmasked, deliberately, to keep recorded 1k results
bit-reproducible. At 8k a window averages twelve documents and only 8.5% of
tokens come from documents that long — unmasked, most of the added context
would be attention across unrelated text, which is not what the extra compute
is meant to buy.

Masking is `segment_ids` in
[`rig/kernels/tpu_flash_attention.py`](../../rig/kernels/tpu_flash_attention.py):
a position may attend only within its own document, subject to causality. The
segment index is derived on-device from the input tokens as a running count of
EOT boundaries, so nothing extra crosses the host boundary.

## What this costs

Traced, not estimated — attention is a minority of FLOPs at these widths, so
eight times the quadratic term is under twice the total:

| tier | FLOPs/token at 1k | at 8k | ratio | attention share |
|---|--:|--:|--:|--:|
| 125M | 710M | 1,371M | 1.93x | 13% → 55% |
| 500M | 3,064M | 5,156M | 1.68x | 10% → 46% |

## Comparing against `reference`

Validation loss is **not** directly comparable between the families. Validation
windows are cut at `seq_len`, so at 8k every scored token gets up to eight
times more context, and this family will post a lower loss for that reason
alone regardless of model quality.

A defensible comparison needs both families evaluated at a **common** context.
Report each at its native context as its own number, and the 8k model
additionally evaluated at 1,024 — RoPE handles the shorter sequence — for the
comparable one. Loss against position within the window is the measurement that
actually shows whether the added range is being used.

## The honest expectation

Only 8.5% of tokens live in documents long enough to fill an 8k window, so a
masked 8k window is largely short documents sitting beside masked-off
neighbours. Masking makes 8k *correct*; it does not make FineWeb *long*. If
this family does not beat `reference` at equal FLOPs, the corpus is the first
place to look, not the recipe.

## What is not built yet

Experts are **replicated**: every device holds all eight and routes only its own
tokens, so there are no expert collectives. That is what the recorded ladder
ran, and it is sufficient while the experts fit in memory.

Expert *parallelism* — sharding experts across devices — is designed but
deliberately not implemented. It becomes necessary when the experts stop
fitting, which is a bigger-model problem, not one this ladder has. The design,
for whoever needs it: under `shard_map` over an `expert` axis, all-gather the
`[E]` counts so every device can derive the full `[P, E]` traffic matrix by
prefix sum; `ragged_all_to_all` the expert-major tokens; block-transpose the
receive buffer, which arrives source-major and expert-minor, into expert-major
order using indices from the counts rather than a sort; `gmm` with
`group_offset` set to the first local expert, which is what that argument is
for; then the inverse `all_to_all` and unsort.

## Decisions this recipe has settled

**Every layer is routed.** All twelve, not alternating. Interleaving dense and
routed layers is common and halves the parameter cost, but it breaks the clean
"active parameters == dense tier" identity, and that identity is what makes the
routed and dense ladders equi-FLOP and therefore comparable. A cheaper model
that cannot be compared is not cheaper for this purpose.

**`E = 8` is fixed across the ladder**, not scaled with width. It is the
simpler experiment and the one the tiers are sized for.

**No shared expert here.** An always-on expert alongside the routed ones is
cheap and usually helps, which is exactly why it does not belong in the
baseline: it adds active FLOPs, so it breaks the equi-FLOP identity against the
dense ladder, and it confounds "did sparsity help" with "did extra dense
capacity help". It is worth measuring — as an ablation forked from this recipe,
with its own arm, so the two questions stay separable.

Use the harness rather than invoking the trainer directly:

```bash
uv run --frozen --no-sync rig run gumbel_local_moe --tier 60m --profile dev -- --local-moe-steps 2
uv run --frozen --no-sync rig run gumbel_local_moe --tier 60m --profile dev -- --local-moe-steps 0
```
