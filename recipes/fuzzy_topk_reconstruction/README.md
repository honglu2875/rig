# Fuzzy Top-K with a train-only reconstruction decoder

This recipe is the reconstruction-only arm of the dead-latent intervention
study. Its deployed transformer is exactly the existing
[`fuzzy_topk_autoencoder`](../fuzzy_topk_autoencoder/) model at `H=16D,
K=4D`. Training adds one separate decoder matrix per MLP block and a normalized
input-reconstruction objective. The extra decoder is absent from evaluation
and is removed from the saved deployment checkpoint.

The sibling [`fuzzy_topk_reconstruction_auxk`](../fuzzy_topk_reconstruction_auxk/)
recipe adds literal residual-reconstruction AuxK to this exact treatment. The
folders are deliberately separate study arms even though their entry scripts
share an implementation.

The original one-seed v6e mechanism screen is predeclared in
[`study-suite-125m-v6e-reconstruction-auxk.json`](study-suite-125m-v6e-reconstruction-auxk.json).
After that preemptible instance disappeared before either treatment gate, the
replacement
[`study-suite-125m-v4-reconstruction-auxk-3seed.json`](study-suite-125m-v4-reconstruction-auxk-3seed.json)
reuses the three already-sealed v4-32 parent runs and launches only the six
missing treatment endpoints. Both suites report additional training work
explicitly rather than shortening the treatments.

## Deployed forward pass

Let `x` be the RMS-normalized input to one MLP, `H=16D`, `K=4D`, and
`G=H/K=4`. The deployed numerical path is unchanged:

```text
z = x W_up + b_up                         # [..., H]
z_group = reshape(z, [..., K, G])
winner = argmax(z_group, axis=G)          # original coordinate within group
a = ReLU(max(z_group, axis=G))            # [..., K]
i = arange(K) * G + winner                # original feature identity
y = sum_j a[j] W_down[i[j], :] + b_down
```

There is no global TopK, stochastic permutation, router, activation
renormalization, balance bias, or frequency-floor surrogate. Groups are fixed
after initialization for the same exchangeability reason documented by the
parent recipe.

## Training-only reconstruction path

Each block owns `W_rec[H,D]`, with no reconstruction output bias. The same
selected values and feature identities reconstruct the stopped normalized MLP
input:

```text
x_target = stop_gradient(x)
x_hat = sum_j a[j] W_rec[i[j], :]
L_rec = sum ||x_hat - x_target||^2
        / sum ||x_target - mean_tokens(x_target)||^2

L_train = cross_entropy + beta_rec * mean_layer(L_rec)
```

Both numerator and denominator are reduced over the complete data-parallel
global batch. The denominator centers each `D` coordinate across tokens. A
`1e-12` floor only protects a zero-energy synthetic batch. The default is
`beta_rec=1.0`.

The custom reverse rule enforces the following gradient contract:

| parameter/input | language-model path | reconstruction path |
|---|---:|---:|
| incoming normalized state `x` | yes | **blocked** |
| `W_up`, `b_up` | yes | yes |
| deployed `W_down`, `b_down` | yes | **blocked** |
| train-only `W_rec` | absent | yes |

Thus reconstruction shapes the feature encoder without turning the auxiliary
target into a shortcut through the residual stream or modifying the deployed
DOWN matrix directly.

## Decoder initialization and optimizer geometry

The parent RNG stream is preserved exactly. For every block, QKV, attention,
UP, and deployed DOWN parameters are drawn in the parent's original order.
`W_rec` consumes no random draw:

```text
W_rec = row_normalize(transpose(W_up))
```

After initialization the matrices are independent parameters. Before Adam,
each `W_rec` row gradient is projected onto the tangent plane of its current
unit direction. After the optimizer update, every row is renormalized to unit
L2 norm. `W_rec` uses the same hidden-weight CompleteP learning-rate and
epsilon multipliers as the other MLP weights, but receives no AdamW weight
decay. Gradient clipping, when enabled, is applied after the tangent
projection.

## Parameters, schedule, evaluation, and checkpoints

The train-only decoder adds exactly

```text
L * H * D = 16 L D^2
```

optimized parameters. This does not change the fixed-TPP ladder denominator,
which remains the deployed parent parameter count. Results report all three
counts explicitly: deployment, train-only, and total optimized parameters.

Validation always calls the ordinary parent fuzzy kernel and never reads
`W_rec`. A retained `checkpoint.npz` is a deployment checkpoint: the
`mlp_reconstruction_w` leaves are stripped before serialization. The packed
training log and result metadata retain the reconstruction trajectory and the
complete implementation contract.

## Physical matrix-FLOP accounting

For `M` tokens in one block, the parent choicewise fuzzy MLP executes
`12MDH` training matrix FLOPs. Reconstruction shares feature scoring and adds
one decoder forward contraction, its value cotangent, and its weight gradient:

```text
forward:   6 M D H
backward: 12 M D H
total:    18 M D H
```

This is 50% more MLP matrix work than the parent treatment. The tracer bills
the named forward and reverse boundaries at these physical counts; it does not
claim an irregular selected-row sparse ideal.

## Default scientific coordinate

All dev and official tiers encode `H=16D, K=4D` directly. The model width and
default TopK are:

| tier | `D` | `H` | `K` | group size |
|---|---:|---:|---:|---:|
| 60m | 384 | 6,144 | 1,536 | 4 |
| 125m | 640 | 10,240 | 2,560 | 4 |
| 250m | 896 | 14,336 | 3,584 | 4 |
| 500m | 1,280 | 20,480 | 5,120 | 4 |
| 1b | 1,792 | 28,672 | 7,168 | 4 |

The YAML default reconstruction settings are:

```yaml
reconstruction:
  coefficient: 1.0
  decoder_unit_norm: true
  auxk:
    mode: none
    coefficient: 0.0
    width_ratio: 0.5
    dead_tokens_threshold: 10000000
```

The inactive AuxK fields remain serialized so this arm and its sibling have
one typed schema. Runtime research overrides are
`--reconstruction-coefficient`, `--fuzzy-auxk-mode`,
`--fuzzy-auxk-coefficient`, `--fuzzy-auxk-width-ratio`, and
`--fuzzy-dead-tokens-threshold`. The resolved `aux_k` must be integral and
divide `K`; reconstruction requires the choicewise backend.

## Predeclared reconstruction-weight sweep

The 125M v4-32 dose-response study in
[`study-suite-125m-v4-reconstruction-beta-sweep-3seed.json`](study-suite-125m-v4-reconstruction-beta-sweep-3seed.json)
reuses the completed `beta_rec=0` parent and `beta_rec=1` endpoints, then runs
only `beta_rec` values `1/4`, `1/16`, and `1/64` at paired seeds 1337--1339.
All other scientific coordinates remain fixed. The sweep retains the full
per-feature activation-frequency observer at cadence 100 and is launched by
[`launch_v4_125m_beta_sweep_3seed.sh`](launch_v4_125m_beta_sweep_3seed.sh).

For parameters shared with the language path, lowering `beta_rec` directly
reduces the reconstruction term in the summed gradient. For the train-only
decoder, whose gradient has no language component, Adam's adaptive
normalization means a smaller raw gradient does not promise a proportionally
smaller parameter update. The decoder is discarded; the study's intended dose
is the reconstruction pressure on the deployed feature encoder.

## Predeclared 20-TPP hero pair

[`study-suite-125m-v4-reconstruction-20tpp-hero.json`](study-suite-125m-v4-reconstruction-20tpp-hero.json)
promotes one positive coefficient only after all three-seed sweep endpoints
finish. Selection minimizes three-seed mean final dev validation loss, with a
smaller-coefficient tie break. The 20-TPP comparison uses held-out seed 1350
for both the 8k dense reference and selected reconstruction treatment.

The dense arm runs its exact 18,838-step fixed-TPP horizon. The sparse arm runs
18,624 steps, matching the dense arm's total deployed active matrix FLOPs to
within 0.002%; its train-only reconstruction decoder's physical work remains
reported rather than being used to shorten the learning horizon. The treatment
retains per-feature activation-frequency captures every 100 steps. The
dependent, fail-closed queue is
[`launch_v4_125m_20tpp_hero.sh`](launch_v4_125m_20tpp_hero.sh).

## Reproduction checks

```bash
JAX_PLATFORMS=cpu .venv/bin/python \
  recipes/fuzzy_topk_reconstruction/train.py --profile smoke --print-plan

JAX_PLATFORMS=cpu .venv/bin/python \
  recipes/fuzzy_topk_reconstruction/train.py \
  --profile smoke --output-dir /tmp/fuzzy-topk-reconstruction-smoke

.venv/bin/pytest -q \
  tests/test_fuzzy_topk_reconstruction.py \
  tests/test_recipe_fuzzy_topk_reconstruction.py
```

Kernel tests compare the custom forward and every input/parameter gradient to
a literal autodiff oracle. Recipe tests pin parent initialization and deployed
forward equivalence, unit-row projection, no-decay classification, parameter
accounting, packed-log layout, and the `18MDH` physical FLOP rule.
