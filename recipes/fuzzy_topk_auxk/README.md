# Paper-inspired dead-feature recovery for fuzzy Top-K

This recipe tests whether two mechanisms from Gao et al., [*Scaling and
evaluating sparse autoencoders*](https://arxiv.org/abs/2406.04093), transfer to
the fuzzy Top-K transformer MLP:

1. initialize each encoder direction parallel to its corresponding decoder
   direction, then train the matrices independently; and
2. give recently dead features an auxiliary training signal.

The [authors' implementation](https://github.com/openai/sparse_autoencoder)
uses tied directions at initialization and an AuxK reconstruction loss over
features that have not fired for roughly ten million tokens. This recipe keeps
those two ideas, but does **not** claim to implement the paper's literal AuxK
loss. A transformer MLP is a residual update, not an autoencoder reconstruction,
so `x - MLP(x)` is not a justified reconstruction residual.

## Three-arm study

The final comparison has one scientific change per treatment arm:

| arm | initialization | training-only dead-feature mechanism | inference |
|---|---|---|---|
| parent fuzzy Top-K | independent UP/DOWN | none | parent fuzzy Top-K |
| frequency floor | independent UP/DOWN | straight-through `b_up` frequency floor | parent fuzzy Top-K |
| paper-inspired | tied UP/DOWN directions at step 0 | reverse-only dead-feature ghost path | parent fuzzy Top-K |

The first two arms are already complete at seeds 1337--1339 for 60M, 125M,
and 250M. They are reused only after an exact coordinate/provenance audit. The
third arm receives nine new runs after its declared systems and mechanism gates.

## Unchanged forward operation

At every training and evaluation step, the numerical MLP output is exactly:

```text
z = x W_up + b_up                         # [..., H]
z_group = reshape(z, [..., K, G])         # G = H / K = 4
winner = argmax(z_group, axis=G)
a = ReLU(max(z_group, axis=G))
i = arange(K) * G + winner
y = sum_j a[j] W_down[i[j], :] + b_down
```

Kernel tests compare the output bit-for-bit with the parent choicewise fuzzy
kernel. Evaluation always calls the parent kernel directly; it carries no dead
state and performs no auxiliary work.

## Tied-direction initialization

For every feature `h`, paper-style initialization sets

```text
W_down[h, :] = W_up[:, h]
```

once, at step zero. The arrays are separate parameters and become untied on the
first update. We retain the CompleteP marginal scale rather than unit-normalize
decoder rows: unit decoder norms belong to the paper's reconstruction
parameterization and would change this transformer's initial residual-update
scale. An unused independent DOWN draw is still consumed, so all attention,
embedding, and other MLP random draws remain identical to the parent seed.

## Dead-feature state

The optimizer stores one saturating `int32` age per layer and stored feature.
After each global batch:

```text
age[h] = 0                         if main fuzzy path activated h
age[h] = min(age[h] + 1, limit)    otherwise
dead[h] = age[h] >= limit
limit = ceil(10,000,000 / global_tokens_per_step)
```

Only positive main-path winners reset the age. Auxiliary activity never makes a
feature appear alive. With the study batch (`16 × 8192 = 131,072` tokens), the
ten-million-token threshold resolves to 77 optimizer steps.

## Zero-forward ghost path

Let `g(x_stop, theta_dead)` be a sparse MLP made only from eligible dead
features, with its input stop-gradient. Conceptually the training forward is

```text
y_train = y_main + alpha * (g - stop_gradient(g))
```

The added value is exactly zero, but reverse mode sends the downstream
language-model cotangent through selected dead rows of `W_down`, `W_up`, and
`b_up`. It sends no auxiliary gradient into the residual-stream input and no
auxiliary gradient into the shared `b_down`. This is a surrogate-gradient
adaptation of AuxK, not a reconstruction objective.

The custom VJP does not materialize `g` and does not recompute `UP(x)`. It
reuses the main preactivations and performs only the required reverse
contractions.

## Fuzzy auxiliary selection

The paper uses approximately `k_aux = D/2`. Here `K=4D`, so the ordinary fuzzy
groups are partitioned into eight rotating cohorts:

```text
k_aux = D / 2
cohort_count = K / k_aux = 8
eligible_groups(step) = arange(step mod 8, K, 8)
```

Within each eligible four-feature group, the largest positive dead score is the
ghost winner. Every group is eligible exactly once per eight steps. This is the
same random-group approximation used by the parent fuzzy Top-K rather than a
global exact top-k over dead features.

For `M` tokens, model width `D`, and dictionary width `H`, the issued MLP matrix
work is:

```text
parent fuzzy training            12 M D H
ghost reverse cohort          +  6 M D H / 8
paper-inspired total             12.75 M D H
```

The nominal MLP matrix premium is therefore 6.25%. TPU wall time, compilation,
and memory still have to pass the measured gate; the formula is not treated as
a throughput result.

## Fixed ladder coordinates

| dense-equivalent tier | L | D | H | main K | aux k | stored parameters | schedule steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| 60M | 11 | 384 | 6,144 | 1,536 | 192 | 97,123,584 | 2,316 |
| 125M | 11 | 640 | 10,240 | 2,560 | 320 | 226,753,280 | 4,656 |
| 250M | 14 | 896 | 14,336 | 3,584 | 448 | 495,053,440 | 9,402 |

All arms use the same v4-32 topology, 8K context, global batch 16, FineWeb
token stream, base learning rate, CompleteP schedule, `H=16D`, `K=4D`, and
choicewise fuzzy kernel.

## Admission protocol

[`ablation-auxk.json`](ablation-auxk.json) is the source of truth.

The internal 60M seed-1350 ablation separates initialization from recovery:

- parent initialization, no ghost;
- tied initialization, no ghost;
- tied initialization plus ghost coefficients `1/128`, `1/32`, and `1/8`.

First, paired 120-step runs test compilation, finite values, dead-state
transitions, and wall time. A candidate must remain within 25% of the paired
parent, with approximately 20% the desired practical ceiling. Admitted cells
then run through step 900 with full-neuron captures every 50 steps. Selection
uses the declared trailing captures and jointly considers persistent death,
positive group activity, winner entropy, validation loss, and gradient
stability. Tied-only is retained as a mechanism ablation even if the combined
cell wins.

The completed systems gate measured:

| cell | train seconds | throughput | overhead vs. parent | validation loss | screen status |
|---|---:|---:|---:|---:|---|
| independent, no ghost | 14.631 | 1.075M tok/s | baseline | 6.2898 | admitted |
| tied, no ghost | 14.537 | 1.082M tok/s | -0.6% | 6.2541 | admitted |
| tied, ghost `1/128` | 16.264 | 0.967M tok/s | +11.2% | 6.2681 | admitted |
| tied, ghost `1/32` | 16.907 | 0.930M tok/s | +15.6% | 6.2675 | admitted |
| tied, ghost `1/8` | 18.364 | 0.856M tok/s | +25.5% | 6.2683 | excluded |

All five bundles pass artifact verification. The `1/8` cell is excluded only
because it narrowly exceeds the predeclared 25% wall-time ceiling; no
scientific metric was used to stop or retry it. The other four cells proceed
to the fixed 900-step mechanism screen.

The mechanism screen completed with the following final-window measurements:

| cell | persistent dead | dead groups | positive groups | winner entropy | validation |
|---|---:|---:|---:|---:|---:|
| independent, no ghost | 35.88% | 5.67% | 9.82% | 0.649 | 4.4266 |
| tied, no ghost | 35.25% | 3.18% | 9.07% | 0.664 | 4.4494 |
| tied, ghost `1/128` | 34.96% | 2.94% | 9.21% | 0.669 | 4.4551 |
| tied, ghost `1/32` | 34.68% | 3.02% | 9.34% | 0.667 | 4.4596 |

Here persistent death means a feature has zero positive activation frequency
in all three predeclared captures at steps 800, 850, and 900. Tied
initialization accounts for most of the small survival change. Incrementally,
`1/128` and `1/32` improve persistent death by only 0.29 and 0.57 percentage
points versus tied-only, while validation regresses by 0.0058 and 0.0102. No
ghost cell therefore satisfies the mature-cell selection rule. The 250M gate
and nine paper-arm ladder runs are deliberately not queued; the completed
screen is retained as a negative result.

Under the protocol, a selected paper arm would next receive one 250M 120-step
feasibility run. Only after that succeeded would seeds 1337--1339 be queued
sequentially at 60M, 125M, and 250M. Because no cell passed the mechanism
screen, that conditional stage was not reached. Scientific intermediate
metrics never trigger retries or early stopping; systems failure pauses the
queue for inspection.

## CLI

The recipe defaults to the paper-style `1/32` arm:

```bash
uv run rig run fuzzy_topk_auxk --cluster v4-32 --profile dev \
  --tier 60m --seed 1350 --name 60m-auxk-c003125-p900-s1350 \
  -- --sparse-layers 11 --sparse-training-steps 2316 --stop-after-step 900 \
  --sparsity-diagnostics-every 50
```

Tied-only ablation:

```bash
uv run rig run fuzzy_topk_auxk --cluster v4-32 --profile dev \
  --tier 60m --seed 1350 --name 60m-tied-only-p900-s1350 \
  -- --sparse-layers 11 --sparse-training-steps 2316 --stop-after-step 900 \
  --fuzzy-auxk-mode none --fuzzy-auxk-coefficient 0 \
  --sparsity-diagnostics-every 50
```

Parent-initialization control additionally passes
`--fuzzy-init-mode independent`. The final paper arm keeps
`--fuzzy-dead-tokens-threshold 10000000`, `--fuzzy-auxk-width-ratio 0.5`, and
the coefficient selected by the versioned screen.
