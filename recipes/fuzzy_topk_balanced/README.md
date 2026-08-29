# Fuzzy Top-K homeostasis

This research fork asks whether training-only homeostasis can prevent the late
feature death observed in [`fuzzy_topk_autoencoder`](../fuzzy_topk_autoencoder/)
without changing its inference-time model. The parent forward operation remains
exactly

```text
z = x W_up + b_up                         # [..., H]
z_group = reshape(z, [..., K, G])         # G = H / K = 4
winner = argmax(z_group, axis=G)
a = ReLU(max(z_group, axis=G))
i = arange(K) * G + winner
y = sum_j a[j] W_down[i[j], :] + b_down
```

Neutral mode is the parent recipe: same parameters, initialization, data order,
optimizer, schedule, decoder identities, and choicewise kernel. Its CPU smoke
training log is byte-identical to the parent and its checkpoint tensors compare
exactly. Evaluation always uses this ordinary forward path, even when training
uses an auxiliary objective.

## Motivation

The sealed per-feature diagnostic ladder shows two distinct late-training
failure modes. Features can lose the within-group competition, and an entire
group can move below the ReLU boundary. The second effect is larger: for the
seed-1337 60M run at step 2,316, 33.0% of features were inactive in the sampled
batch but only 2.3% had never won; the average token had a positive winner in
about 144 of 1,536 groups. At 500M, block 1 reached about 84% dead features late
in training. The original three-seed trailing-window dead fractions were 34.3%,
35.8%, 33.9%, and 36.3% at the 60M, 125M, 250M, and 500M tiers.

This fork therefore separates **choice balance** from **group survival** rather
than treating all zero activations as one phenomenon.

## Training-only objectives

For token `t`, fixed group `g`, and one of its four feature choices `c`, let

```text
p[t,g,c] = softmax(z[t,g,:] / temperature)[c]
load[g,c] = mean_t 1[argmax_c z[t,g,c] = c]       # stop-gradient
importance[g,c] = mean_t p[t,g,c]
```

Two interchangeable choice-balance objectives are implemented:

```text
L_switch = mean_g G * sum_c load[g,c] * importance[g,c]
L_importance = mean_g (G * sum_c importance[g,c]^2 - 1)
```

`L_switch` is the fixed-group analogue of the Switch Transformer auxiliary
loss. Its minimum is one. `L_importance` is a fully smooth population
concentration penalty with minimum zero. Neither objective forces every token
to use every feature; both ask different tokens in the global batch to
distribute their choices across the four stored features.

The orthogonal group-survival objective is

```text
L_alive = mean_{t,g} ReLU(alive_margin - max_c z[t,g,c])^2
```

It is exactly zero for groups whose winner clears the requested margin. With a
zero margin it acts only when the whole group would be erased by ReLU. A run may
use either choice objective, the survival objective, or both:

```text
L_train = L_CE + balance_coefficient * L_choice
                 + alive_coefficient * L_alive
```

The logged `loss` remains unpolluted cross entropy. All three raw objectives are
logged model-wide and per layer; the existing forward-only full-neuron observer
can simultaneously record winner and activation frequencies.

## Efficient differentiated path

The balanced custom VJP shares feature scoring with the ordinary forward pass,
recomputes it once during reverse mode, and merges every auxiliary cotangent
into the existing choicewise `dX`, `dW_up`, and `db_up` contractions. The
decoder forward and gradients are unchanged. For `M` tokens, its physical MLP
matrix work is

```text
forward + backward = 14 M D H
neutral parent      = 12 M D H
```

so the intended matrix-work premium is 16.7%, not another dense auxiliary
forward. Kernel tests compare outputs and all five operand gradients with a
literal dense-hidden oracle. A TPU timing gate still decides whether the
compiled implementation is mature enough for quality runs.

## Fixed comparison coordinates

All treatments retain the successful fuzzy arm's `H=16D`, `K=4D`, group-size
four coordinates and its matched-active-FLOP schedule. Only the training
objective differs.

| dense-equivalent tier | L | D | H | K | group choices | stored parameters | schedule steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| 60M | 11 | 384 | 6,144 | 1,536 | 4 | 97,123,584 | 2,316 |
| 125M | 11 | 640 | 10,240 | 2,560 | 4 | 226,753,280 | 4,656 |
| 250M | 14 | 896 | 14,336 | 3,584 | 4 | 495,053,440 | 9,402 |
| 500M | 16 | 1,280 | 20,480 | 5,120 | 4 | 1,072,968,960 | 19,561 |

| arm | forward/inference | extra training signal | issued MLP matrix work |
|---|---|---|---:|
| parent fuzzy control | grouped fuzzy Top-K | none | `12MDH` |
| Switch balance | identical | hard-load x soft-importance | `14MDH` |
| importance balance | identical | smooth importance concentration | `14MDH` |
| alive margin | identical | negative group-maximum hinge | `14MDH` |
| combined | identical | one choice loss + alive margin | `14MDH` |

The existing uninstrumented and per-feature parent runs remain valid controls;
the neutral implementation is rerun only for systems and trajectory gates.

## Experiment protocol

The versioned [`ablation-homeostasis.json`](ablation-homeostasis.json) is the
source of truth. Development first screens one paired 60M seed through step 900,
which covers the observed onset of late death. It brackets objective family,
coefficient, and alive margin without selecting on a two-step smoke loss. Every
prototype records full-neuron statistics every 50 steps.

A treatment advances only if it:

1. completes the standard v4-32 preflights and a 120-step timing/stability gate;
2. adds no more than 25% measured training time over the paired neutral gate;
3. materially improves trailing feature/group activity by step 900 without a
   catastrophic loss or gradient regression; and
4. is chosen from the declared grid, never from unrecorded retries.

The mature treatment is then run for paired seeds 1337--1339 at 60M, 125M, and
250M, with the original fuzzy per-feature observer at cadence 10. The complete
schedule, validation contract, and data order stay fixed. A systems failure
stops the sequential queue for inspection; scientific intermediate metrics do
not stop or retry a run. A 500M extension is queued only if the smaller ladder
finishes within the available accelerator window or remains useful as a
separate continuation.

## CLI

```bash
uv run rig run fuzzy_topk_balanced --cluster v4-32 --profile dev \
  --tier 60m --seed 1337 --name 60m-switch-c001-s1337 \
  --recipe-arg=--sparse-layers --recipe-arg=11 \
  --recipe-arg=--sparse-top-k --recipe-arg=1536 \
  --recipe-arg=--sparse-training-steps --recipe-arg=2316 \
  --recipe-arg=--fuzzy-balance-mode --recipe-arg=switch \
  --recipe-arg=--fuzzy-balance-coefficient --recipe-arg=0.01 \
  --recipe-arg=--sparsity-diagnostics-every --recipe-arg=10
```

`--fuzzy-balance-mode` is `none`, `switch`, or `importance`. Coefficients and
temperature must be positive when choice balance is enabled. The alive
coefficient and margin are independent nonnegative values. All homeostasis
objectives require the `choicewise` sparse MLP backend.
