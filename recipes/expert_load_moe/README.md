# expert_load_moe — load-scaled optimizer experiment

A research fork of [`reference_moe`](../reference_moe/) that changes only how
the four parameter tensors belonging to each expert are optimized. Architecture,
routing, initialization, data, schedule, base learning rate, AdamW settings,
and the router auxiliary loss remain those of the MoE baseline.

Like the baseline, this fork has three complete configuration documents:
`config.yaml` for official runs, `dev.yaml` for development runs, and
`smoke.yaml` for the routed CPU check. `--profile` only selects the file.

| | `reference_moe` | `expert_load_moe` |
|---|---|---|
| context, model, router | top-2 of 8, 8k default | identical |
| base optimizer | Complete(d)P-flavoured AdamW | identical |
| expert adjustment | none | current-load gradient or update scaling |
| default adjustment | — | update mode, strength 0.5 |

## Scaling rule

For layer \(l\), let \(p_{l,e}\) be expert \(e\)'s fraction of the current
global batch's hard top-k assignments. The balanced reference is \(1/E\), so
the full square-root batch/LR rule gives

```text
full_scale[l, e] = sqrt(p[l, e] / (1 / E)) = sqrt(E * p[l, e])
scale[l, e]      = 1 + c * (full_scale[l, e] - 1)
```

`c` is `expert_load_scaling_strength` in `[0, 1]`. At `c=0` the fork takes the
baseline optimizer path exactly. At the default `c=0.5`, an unused expert still
has scale 0.5; at `c=1`, the literal square-root rule may give an unused expert
zero update. Balanced experts always have scale 1. The load is already averaged
over the data mesh and is explicitly stop-gradient.

The two modes deliberately answer different questions:

- `update` multiplies the normalized Adam update and decoupled weight decay.
  It is a literal per-expert learning-rate multiplier and has a persistent
  effect even when load is stable.
- `gradient` multiplies the post-global-clip gradient before both Adam moments.
  A stable multiplier largely cancels between the first and second moments;
  its intended effect is the transient produced when an expert's load changes.

Only `expert_up_w`, `expert_up_b`, `expert_down_w`, and `expert_down_b` are
scaled. Router, attention, norms, and embeddings retain scale 1. The complete
per-layer load remains in the riglog, so every applied multiplier is derivable
afterward without adding another high-frequency metric.

This is not another load-balancing mechanism. In particular, giving an already
busy expert a larger update could create positive feedback through the router.
The existing auxiliary loss remains active; this fork measures whether matching
expert optimizer scale to realized expert batch size helps despite that risk.

## Per-expert diagnostics

Sparse diagnostic captures retain the baseline model-wide and per-block scopes,
then add one nested scope for every routed expert:

```text
block[i]/expert[j]/{param,grad,update}.{statistic}
```

Each expert scope contains exactly its up/down weights and biases—the same four
leaves the intervention scales. Router weights remain in the parent block scope;
mixing them into an expert scope would make its update statistics describe both
scaled and unscaled parameters. Every family records L1 norm, L2 norm, mean,
standard deviation, third and fourth centered moments, and p01/p10/p50/p90/p99.
`grad` remains the raw objective gradient and `update` is the actual signed
parameter delta, so the two optimizer modes can be distinguished directly.

Percentiles use at most 2,048 deterministic midpoint samples spread over the
scope's logical flattened parameter sequence. Scopes with at most 2,048 values
are exact. Sampling avoids sorting millions of values for each expert on every
diagnostic step; its method and cap are written into implementation provenance.
The existing diagnostics cadence controls these captures, while the exact
per-step router-load vector stays in `training.riglog` as before.

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

Use the harness rather than invoking the trainer directly. CLI overrides make
matched arms possible without editing the frozen config:

```bash
uv run --frozen --no-sync rig run expert_load_moe --tier 125m --profile dev \
  -- \
  --expert-load-scaling-mode update --expert-load-scaling-strength 0.5
uv run --frozen --no-sync rig run expert_load_moe --tier 125m --profile dev \
  -- \
  --expert-load-scaling-mode gradient --expert-load-scaling-strength 1.0
```
