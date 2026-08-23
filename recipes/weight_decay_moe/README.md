# weight_decay_moe — AdamW weight-decay sweep

This scientific fork keeps `reference_moe` unchanged except for one explicit
recipe-local `--weight-decay` override. Omitting the option preserves the YAML
default of `0.1`; the override changes the base AdamW coefficient before the
existing Complete(d)P tensor and token-horizon multipliers are applied.

## Initial study contract

| coordinate | values |
|---|---|
| tier | 60M, 125M |
| base weight decay | 0, 0.03, 0.1, 0.3 |
| seeds | 1337, 1338, 1339 |
| fixed training setup | 8k context, 5 TPP, batch 16, base LR 0.00390625 |
| fixed MoE setup | top-2 of 8 experts, router auxiliary coefficient 0.01 |
| machine | TPU v4-32 |
| checkpoints | none |

The grid contains 24 fresh runs. In particular, the `0.1` controls are rerun
under the same commit and logging schema rather than imported from an older
study.

### 125M upper-decay extension

The initial grid found that `0.3` improved 125M validation loss for all three
seeds and was still the best value at the upper boundary. A second phase keeps
the 125M setup unchanged and evaluates base weight decay `0.4`, `0.5`, `0.6`,
and `0.8`, again at seeds 1337–1339. These 12 fresh runs reuse the initial
`0.3` cell as their lower boundary rather than rerunning it.

## Sealed result

The completed study has 36 verified runs. At 125M, base coefficient `0.3`
wins all three paired seeds and averages **3.567633 ± 0.007781** validation
loss (mean ± sample SD), improving on the `0.1` default by 0.015184 nats.
The minimum is broad: `0.4` and `0.5` trail by 0.005930 and 0.004471 nats;
`0.6` has turned upward and `0.8` is effectively back at no decay.

At 60M, the raw mean chooses `0.1` (3.925148 ± 0.003116), but the apparent
reversal is dominated by the `0.3`, seed-1338 run, which has a gradient-norm
spike of 20.41 at step 57 and finishes 0.046494 nats behind its paired `0.1`
run. The other two paired deltas are -0.013089 and +0.002188. Treat `0.3` as
the 125M-specific working choice, retain `0.1` as the cross-tier default, and
resolve the smaller tier with more seeds around `0.1`–`0.3` if needed.

See the static [findings report](../../docs/reports/moe-weight-decay.html) and
the full-resolution
[`moe-weight-decay`](https://huggingface.co/datasets/quintic/rig-logs/tree/main/moe-weight-decay)
archive. The archive card records exact tables, hashes, commands, and the
project-local CompleteP/Complete(d)P scaling interpretation.

## Inherited MoE design

A fork of [`reference`](../reference/) that replaces every dense MLP with an
eight-expert, top-2 routed MLP. The fork keeps the same named context contracts;
it defaults to `8k`, while `--context 1k` selects the short-context contract on
demand.

Like `reference`, this fork has three complete configuration documents:
`config.yaml` for official runs, `dev.yaml` for development runs, and
`smoke.yaml` for the routed CPU check. `--profile` only selects the file.

| | `reference` | `reference_moe` |
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
uv run --frozen --no-sync rig run weight_decay_moe --tier 60m --profile dev \
  --context 8k --tokens-per-parameter 5 --batch-size 16 \
  --base-learning-rate 0.00390625 --seed 1337 \
  --checkpoint-policy none --name 60m-wd0p03-s1337 -- \
  --weight-decay 0.03
```
