# Leave-one-out infinigram distillation

This recipe changes only the training objective of the dense reference GPT. Its
baseline (`--infinigram-weight 0`) takes the original cross-entropy path and
does not open or query an index. A treatment minimizes

```text
a * CE(ground_truth, model) + b * CE(infinigram, model),  a + b = 1.
```

The normalization constraint holds the loss and optimizer scale constant while
we tune the mixture. The default is `a=1, b=0`; passing
`--infinigram-weight 0.25` implicitly sets `a=0.75`. An explicit
`--ground-truth-weight` is accepted only when the two coefficients sum to one.

The infinigram distribution is not transferred as a 50,257-wide vector. For
every training token, a host-side suffix-array query draws `K` continuations
with replacement, `t_j ~ P_infini(. | context)`. Then

```text
E_t[-1/K sum_j log P_model(t_j | context)] = CE(P_infini, P_model),
```

so their average is an unbiased Monte Carlo estimator of the full teacher
cross-entropy, with sampling variance decreasing approximately as `1/K`. The
sampler performs the suffix search and continuation count once per position;
only the cheap draws repeat. The tiled output head also computes the shared
log-normalizer once:

```text
(a + b) * logsumexp(logits) - a * logits[y]
    - (b/K) * sum_j logits[t_j]
```

It neither materializes teacher logits nor repeats the vocabulary-wide output
projection. `--infinigram-samples` selects K and defaults to 1 for exact
backward compatibility. Canonical validation remains ground-truth-only.

For K greater than one, the custom VJP keeps the vocabulary-wide softmax and
ground-truth correction in the existing tiled matmul loop, then applies the K
teacher corrections sparsely in chunks of eight gathered embedding rows. This
is O(KD), rather than comparing K target ids with all V vocabulary rows. The
largest live teacher temporary is `[positions, 8, D]`; no `[positions, K, V]`
or `[positions, K, D]` tensor is formed. K=1 retains the previous tiled path.

## Why leave one occurrence out

The teacher index contains the same FineWeb training split. A vanilla
infinigram query can find the exact occurrence currently being trained and use
an arbitrarily long suffix to return its next token. That is target leakage,
not useful distillation.

The included ngram patch therefore removes one occurrence of the observed
continuation before sampling. If no continuation remains, it backs off to a
shorter suffix until the adjusted distribution is nonempty. Context resets at
GPT-2 EOT (`50256`), with the EOT itself retained as the first token of the new
document. Shard-tail suffixes without a successor do not contribute mass. The
sampler is exact for this adjusted distribution and deterministic from
`(run seed, optimizer step, JAX process index)`.

## Reproduce the ngram dependency

The on-disk format and builder start from
[`honglu2875/ngram`](https://github.com/honglu2875/ngram) revision
`2a73b5ffbe852718dbd4e01ee6abafeb1628c5a7`. Apply the checked-in patch and
install it into the rig environment:

```bash
git clone https://github.com/honglu2875/ngram /tmp/ngram
git -C /tmp/ngram checkout 2a73b5ffbe852718dbd4e01ee6abafeb1628c5a7
git -C /tmp/ngram am "$PWD/recipes/infinigram_distillation/patches/0001-Add-leave-one-out-infinigram-distillation-sampler.patch"
git -C /tmp/ngram am "$PWD/recipes/infinigram_distillation/patches/0002-Add-shared-query-K-sample-leave-one-out-targets.patch"
uv pip install --python .venv/bin/python --editable /tmp/ngram
```

The first patch adds the batched leave-one-out sampler and removes optional
tokenizer imports from token-ID-only build/query paths. The second adds K draws
per shared query while retaining the original one-sample API. Its 69-test
upstream suite passes. Every training result records both patches, the Python
wrapper, and compiled extension SHA-256 digests. Existing indexes remain valid:
the second patch changes querying, not the on-disk format or builder.

## Build and query the 8B index

The prior 125M long-context hero used the prepared FineWeb-8B corpus: 7.9B
training tokens plus a separate 100M-token validation shard. Only the 79
training shards belong in the teacher. The recipe utility validates every
source shard against the pinned 8B manifest, builds a resumable two-shard
suffix-array index, and writes `provenance.json`:

```bash
.venv/bin/python recipes/infinigram_distillation/index_tool.py build \
  --data-root /dev/shm/.speedrun-cache/fineweb-scaled/8B \
  --output /home/cubic27/infinigram/fineweb-8b-gpt2-sa-v1 \
  --cpus 240 --mem 300 --shard-tokens 4G
```

The finished index is about 45 GiB. Each TPU VM needs an identical local copy
at the same path; queries are memory-mapped and do not use TPU compute. Building
is resumable, so rerunning the same command completes only missing shards.

Inspect it or query a GPT-2 token-id context without loading a tokenizer:

```bash
.venv/bin/python recipes/infinigram_distillation/index_tool.py info \
  /home/cubic27/infinigram/fineweb-8b-gpt2-sa-v1
.venv/bin/python recipes/infinigram_distillation/index_tool.py query \
  /home/cubic27/infinigram/fineweb-8b-gpt2-sa-v1 464 329 --top 10
```

Training refuses an incomplete/unattested index, the wrong dataset manifest,
wrong shard inventory, tokenizer, boundary token, vocabulary, or token count.

## Coefficient and sample-count studies

Use the 125M, 8k-context, 5-TPP setting to screen `b in {0.10, 0.25, 0.50}`
at seed 1337 against the existing exact dense control. Then run seeds 1338 and
1339 only for the selected coefficient. This staged design tunes the mixture
without spending nine treatment runs before we know its useful range. A final
20-TPP single-seed treatment can compare with the existing 125M hero only after
the development sweep selects `b`.

That K=1 screen selected `b=0.10` among the treatment settings, while larger
teacher weights were progressively worse and no K=1 treatment beat the dense
control. The variance-reduction follow-up therefore uses the paired grid
`b in {0.025, 0.05, 0.10}` by `K in {16, 32, 64}` at seed 1337. For a fixed
run seed, the K=16 samples are the prefix of K=32 and K=64, reducing irrelevant
Monte Carlo differences between columns. This grid tests whether the prior
teacher signal was too noisy while concentrating weight below the best K=1
coefficient. It reuses the completed dense and K=1 controls.

The v4-32 throughput gate used the exact 125M, 8k-context shape for 100 steps.
Against 713k token/s at K=1, the accepted sparse-gradient path measured 672k
at K=16, 625–640k at K=32, and 600k at K=64. Thus the throughput reductions
were about 6%, 10–12%, and 16%, respectively. Two independent K=32 launches
produced byte-identical training and diagnostic logs. The frozen launch
contract is in
[`study-k-sample-grid-125m-v1.json`](study-k-sample-grid-125m-v1.json).

All nine K-by-weight cells completed at 5 TPP. Canonical validation loss was:

| teacher samples K | b=0.025 | b=0.05 | b=0.10 | full-run throughput |
|---:|---:|---:|---:|---:|
| 16 | **3.653455** | 3.670624 | 3.707323 | 701–703k token/s |
| 32 | 3.665286 | 3.696929 | 3.749261 | 669–672k token/s |
| 64 | 3.673222 | 3.706025 | 3.761992 | 620–622k token/s |

The paired dense seed-1337 control is 3.641529. The best new cell, K=16 and
b=0.025, is 0.011926 nats worse. At every measured teacher weight, increasing
K worsened the endpoint; at every K, increasing teacher weight also worsened
it. The denser Monte Carlo estimate therefore did not rescue this teacher
objective. The prior K=1, b=0.10 endpoint of 3.641004 is only 0.000525 below
the paired dense run, far inside the dense three-seed spread and not evidence
of an improvement.

The learning-rate follow-up fixed K=16 and crossed the same three teacher
weights with 1.25x, 1.5x, and 2x the 2^-8 base-LR anchor, while reusing the
completed 1x row. Its canonical validation surface was:

| base-LR multiplier | b=0.025 | b=0.05 | b=0.10 |
|---:|---:|---:|---:|
| **1.0x** | **3.653455** | **3.670624** | **3.707323** |
| 1.25x | 3.660932 | 3.679066 | 3.717053 |
| 1.5x | 3.716397 | 3.725557 | 3.739457 |
| 2x | 3.724212 | 3.753610 | 3.801402 |

Every LR increase worsened every teacher-weight cell. Full-run throughput
remained 699–704k token/s, so the degradation is not a systems artifact. The
frozen contract is in
[`study-k16-learning-rate-grid-125m-v1.json`](study-k16-learning-rate-grid-125m-v1.json).

The scaling hero returns to the only development setting that reached the
paired dense control: K=1 and b=0.10. It trains a 500M, 8k-context model for
20 TPP at seed 1337, paired with a newly trained b=0 control under the exact
same contract. Existing 500M heroes are not reused because they use 1k context
and batch 128 rather than 8k context, document masking, and batch 16. The
treatment runs first, followed by the control. Checkpoints, activation
distributions, and sparsity snapshots are disabled; only compact standard
training, scalar diagnostic, validation, metrics, and provenance artifacts are
written. The frozen protocol is in
[`study-k1-500m-20tpp-hero-v1.json`](study-k1-500m-20tpp-hero-v1.json).

```bash
uv run --frozen --no-sync rig run infinigram_distillation \
  --profile dev --tier 125m --context 8k --tokens-per-parameter 5 --seed 1337 \
  --name infinigram-b025-s1337 --checkpoint-policy none -- \
  --infinigram-weight 0.25 \
  --infinigram-samples 32 \
  --infinigram-index /home/cubic27/infinigram/fineweb-8b-gpt2-sa-v1 \
  --infinigram-threads 30
```

Besides ordinary losses and throughput, `metrics.json` records position-query
and teacher-sample rates separately, query critical-path seconds, matching
suffix lengths, the adjusted teacher probability of the ground-truth token,
and how often sampled teacher targets equal ground truth.

## Dense model family

`train.py` is a readable pure-JAX GPT family, not a claimed record. Model,
fixed-TPP CompleteP-hybrid AdamW, batching, sharding, timing, evaluation, and checkpoint logic
remain visible in this one entry file. Three strict standalone documents select
the execution contract: `config.yaml` is official, `dev.yaml` is development,
and `smoke.yaml` is the tiny CPU wiring test. Each document is complete; no YAML
inheritance or runtime profile overlay is involved.

The non-smoke tiers are:

| Tier | Layers | Width | Heads | Exact parameters |
|---|---:|---:|---:|---:|
| 60M | 12 | 384 | 6 | 59,918,208 |
| 125M | 12 | 640 | 10 | 123,456,640 |
| 250M | 16 | 896 | 14 | 244,444,032 |
| 500M | 19 | 1,280 | 20 | 502,602,240 |
| 1B | 21 | 1,792 | 28 | 989,943,808 |

Every tier uses 64-wide heads, base-10,000 RoPE, pre-RMSNorm, a 4× GELU
MLP, and untied input/output embeddings. The family is identified as
`reference-gpt-v3-family`. Width, depth, initialization, residual, attention,
per-tensor AdamW and recipe-local batch/data scaling are specified in
[the fixed-TPP parameterization contract](../../docs/COMPLETEP.md).

`make run` selects 125M by default. Use `make run TIER=60m`,
`TIER=250m`, `TIER=500m`, or `TIER=1b` to select another tier. The official
profile trains for approximately 20 tokens per parameter, rounded to a complete
global step. The dev profile uses 5 TPP. `--stop-after-step` reproduces a prefix
without changing either schedule; smoke keeps a tiny standard-parameterized CPU
wiring test.

## Data parallelism

Dev and official training use `shuffled_epochs`. A deterministic keyed
permutation traverses non-overlapping windows without allocating a global index
array. Every JAX process constructs the same global order and consumes only its
rank-local slice, so hosts do not duplicate examples within a global batch.
The rank comes from `jax.process_index()` after `jax.distributed.initialize()`,
not from an assumed `RANK` environment variable or hostname suffix. Model and
optimizer state remain replicated; this is data sharding, not model sharding.

The stream starts a newly seeded shuffled epoch if a requested horizon exceeds
the prepared distinct-token capacity. Prepare a corpus large enough for the
horizon you want when a study needs to stay no-replacement.

## Context presets and document boundaries

The dense implementation is shared. `reference` defaults to the established
1k baseline; `--context 8k` selects the long-context contract:

| preset | sequence length | reference batch | document masking | tokens/step |
|---|---:|---:|---|---:|
| `1k` (default) | 1,024 | 128 | off | 131,072 |
| `8k` | 8,192 | 16 | on | 131,072 |

The batch is part of the preset because it is the recipe-local anchor used by
`m_B`. Holding it at 128 for 8k would multiply tokens per optimizer step by
eight and divide the fixed-TPP schedule by eight. An explicit `--batch-size`
still overrides the selected preset's anchor for a batch study.

The corpus is a flat token stream — whole FineWeb documents, each prefixed by
the GPT-2 EOT token `50256`, concatenated with no offset index. Windows are cut
live at arbitrary positions (`shard[start : start + seq_len + 1]`), so a window
can begin mid-document and run through several.

At the default 1k preset, attention is causal only: a token can attend across
an EOT boundary into the preceding document. This preserves every recorded 1k
baseline exactly.

Measured on a 100M-token shard (143,857 documents, median length 405 tokens):

| | at `seq_len` 1024 |
|---|--:|
| documents a random window spans | ~1.5 |
| documents that fit whole (< 1024 tokens) | 83.7% |
| **tokens living in documents ≥ 1024 tokens** | **53.0%** |

So most *documents* are shorter than the window and sit inside it entirely,
while most *tokens* come from documents long enough that a 1024-token window
truncates them. Either way the cross-document surface is small: about one
boundary per window.

Two reasons this stays off:

- **Bit-identical reproducibility.** Every result in
  [the transfer note](../../docs/HYPERPARAMETER_TRANSFER.md) was measured
  without masking. Enabling it would silently redefine the baseline and make
  new 1k runs incomparable with the recorded ones — the 250M cells reproduced
  bit-identically across a refactor and a package rename, and that property is
  worth more than a marginal correction.
- **The baseline should stay basic.** 1024 tokens is a short context; masking
  buys little at ~1.5 documents per window and adds a data-dependent term to
  the attention mask.

It is not free of consequence, and the honest statement is that the 1k preset
trains with a small amount of cross-document attention.

The `8k` preset **does** mask, because an 8,192-token window spans about 11.8
documents and only 8.5% of tokens live in documents that long. Without masking,
most of the added range would be attention across unrelated text. The preset
derives segment ids on-device from EOT boundaries; implementation is in
[`rig/kernels/tpu_flash_attention.py`](../../rig/kernels/tpu_flash_attention.py).

Native validation losses at 1k and 8k are not directly comparable: each token
in the 8k evaluation may receive much more context. Report native loss for each
contract, and evaluate the 8k model at 1k as well when the question requires a
common-context comparison.

## Optimizer and kernels

The family uses the fixed-TPP CompleteP hybrid with α=1, PyTorch-form AdamW,
10% warmup, cosine
decay to 10% of peak, and no global gradient clipping. The custom trainable
Pallas attention backend and tiled output cross entropy are selected for dev
and official TPU profiles. Smoke uses dense FP32 kernels.

For non-dense attention, the trainer resolves a static ten-field tile plan
before compiling the real step. Resolution checks an exact runtime-fingerprinted
cache, a source-pinned shipped entry, and then a deterministic shape heuristic.
An explicit synthetic autotuner is available; it never reads real data or runs
inside timed training. See [the kernel notes](../../docs/KERNELS.md).

## Artifacts

Every accepted run writes:

- `training.riglog`: every optimizer step's train loss, effective global LR, and
  gradient norm;
- `validation.csv`: deterministic probes and canonical final validation;
- `diagnostics.riglog`: sparse parameter, gradient, and update statistics for
  every supported model scope and statistic;
- `checkpoint.npz`, except for explicit development study runs using
  `--checkpoint-policy none`;
- `metrics.json`, with the resolved tier, exact parameter count, parameterization
  multipliers, data-sharding rule, system topology, and result protocol.

Both logs are packed binary, not CSV: a header naming each column by its
permanent id from `rig/metrics.py`, then fixed-width `float32` records. A 500M
run's diagnostics are 6.4 MB instead of 144 MB, and the report reads them in
milliseconds. Fixed stride also makes the file append-only, so a preempted run
keeps every sample already written. Read one with `rig.logpack.read_log`, which
returns the column table plus one `(samples x columns)` array; the byte
layout and the metric-id registry are in
[docs/RIGLOG_FORMAT.md](../../docs/RIGLOG_FORMAT.md).

Curves accumulate on device and move to the host after synchronized training,
so per-step capture does not add a host synchronization. Dev and official
runs enable all diagnostics by default, every 10 and 500 steps respectively
(plus the first and final steps); smoke runs keep them disabled. Official probes
run every 500 steps and count inside `train_seconds`; canonical final FineWeb
and Fresh10 evaluation run outside it.

`--checkpoint-policy none` is restricted to development research. It preserves
metrics, curves, immutable records, and re-verification while never writing
hundreds of megabytes of weights at every sweep point. `qualifying` keeps them
only at or below the target loss; `always` keeps them regardless. Official runs
still require a checkpoint.

Use the harness rather than invoking `train.py` directly:

```bash
uv run --frozen --no-sync rig run infinigram_distillation --profile smoke
uv run --frozen --no-sync rig run infinigram_distillation --profile official
uv run --frozen --no-sync rig run infinigram_distillation \
  --context 8k --tier 60m --profile dev
```
