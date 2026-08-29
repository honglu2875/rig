# Fuzzy Top-K reconstruction decoder with literal AuxK

This recipe is the reconstruction+AuxK arm of the dead-latent intervention
study. It contains the complete train-only decoder treatment documented by
[`fuzzy_topk_reconstruction`](../fuzzy_topk_reconstruction/), then adds a
paper-inspired auxiliary reconstruction of the main decoder's residual using
features that have been inactive for a configured token age.

The deployed transformer is still exactly the parent fuzzy `H=16D, K=4D`
model. Neither reconstruction output enters the transformer residual stream,
and the separate decoder is removed from evaluation and saved deployment
checkpoints.

The paired parent/reconstruction/reconstruction+AuxK studies are predeclared
in the sibling recipe's one-seed
[`study-suite-125m-v6e-reconstruction-auxk.json`](../fuzzy_topk_reconstruction/study-suite-125m-v6e-reconstruction-auxk.json)
and three-seed v4-32
[`study-suite-125m-v4-reconstruction-auxk-3seed.json`](../fuzzy_topk_reconstruction/study-suite-125m-v4-reconstruction-auxk-3seed.json).

## Main path and reconstruction objective

For the RMS-normalized block input `x`, fixed-group fuzzy TopK selects one of
four stored features in each of `K=4D` groups:

```text
z = x W_up + b_up
(a, i) = GroupedTopKReLU(z, groups=K, choices=4)
y = sum_j a[j] W_down[i[j], :] + b_down

x_target = stop_gradient(x)
x_hat = sum_j a[j] W_rec[i[j], :]
L_rec = NMSE(x_hat, x_target)
```

`W_rec[H,D]` is a separate no-bias decoder initialized as the row-normalized
transpose of `W_up`. Its gradient is projected row-wise onto the unit sphere's
tangent plane before Adam, it receives no weight decay, and its rows are
renormalized after every update. Reconstruction gradients update `W_up`,
`b_up`, and `W_rec`; they do not update incoming `x`, deployed `W_down`, or
`b_down`.

## Dead-feature age

Every training step returns the main selector's positive-activation count for
all `L*H` features, reduced over the complete global batch. A feature's int32
age is then updated as:

```text
age[f] = 0                              if main_count[f] > 0
age[f] = min(age[f] + 1, dead_after)    otherwise
dead[f] = age[f] >= dead_after

dead_after = ceil(dead_tokens_threshold / (global_batch * sequence_length))
```

The default threshold is 10,000,000 tokens. At the ladder coordinate
`batch=16, sequence=8192`, this is 77 optimizer steps. The mask used by a step
comes from ages before that step; main activity from the current batch updates
the state for the next step. An AuxK activation never resets age.

## Fuzzy AuxK selector

The literal SAE algorithm would take exact TopK over all currently dead
preactivations. This recipe preserves the project's cheap fixed-group
approximation. With `K=4D` and default `k_aux=D/2`, `K/k_aux=8` cohorts cover
the outer groups:

```text
cohort = previous_optimizer_step mod 8
group_ids = cohort + 8 * arange(k_aux)

for each selected outer group:
    mask its four candidates to currently dead features
    choose the largest masked preactivation
    aux_a = ReLU(chosen score), or zero if no candidate is dead
```

One rotating cohort therefore visits `D/2` outer groups per token and step;
all eight cohorts are visited over eight consecutive steps. Selected rows
retain their original feature identities. The selector is deterministic given
the optimizer step and dead-age state.

## Literal residual-reconstruction loss

AuxK decodes with the same `W_rec` and reconstructs a stopped copy of the main
reconstruction residual:

```text
r_target = stop_gradient(x - x_hat)
r_aux = sum_j aux_a[j] W_rec[aux_i[j], :]
L_aux = sum ||r_aux - r_target||^2
        / sum ||r_target - mean_tokens(r_target)||^2

L_train = cross_entropy
        + beta_rec * mean_layer(L_rec)
        + beta_aux * mean_layer(L_aux)
```

Reductions and centered denominators cover the full data-parallel global
batch. Defaults are `beta_rec=1.0` and `beta_aux=1/32`. AuxK gradients update
only the selected encoder columns, their biases, and selected rows of `W_rec`.
They are blocked from the incoming residual stream and deployed DOWN matrix.
This is a real positive forward loss, unlike the earlier zero-forward ghost
surrogate.

## Logging

Every optimizer step stores model-wide and per-layer values for:

- reconstruction NMSE;
- AuxK NMSE;
- fraction of the `k_aux` slots with a positive auxiliary activation;
- fraction of features active on the main path in that global batch; and
- fraction whose tracked age is at the dead threshold.

The ordinary full-feature `fuzzy_sparsity.rigvec` observer remains available at
its configured cadence. It is a separate forward-only executable and does not
alter the optimizer graph.

## Parameters, schedule, and physical FLOPs

Like the reconstruction-only arm, training optimizes an extra `LHD=16LD^2`
decoder parameters while fixed-TPP duration remains anchored to the deployed
parent parameter count. Results distinguish deployment, train-only, and total
optimized counts. Validation ignores `W_rec`, and checkpoint serialization
strips it.

Let `r=k_aux/K`. AuxK adds `2rMDH` forward decoder work and `6rMDH` reverse
work to reconstruction training:

```text
reconstruction only: 18 M D H
reconstruction+AuxK: (18 + 8r) M D H
default r = 1/8:      19 M D H
```

At the default coordinate, AuxK is therefore 5.56% additional MLP matrix work
over reconstruction-only and 58.33% over the parent `12MDH` fuzzy MLP. These
are issued regular-contraction counts, not an ideal gathered sparse estimate.

## Default typed configuration

All tiers directly encode `H=16D, K=4D`. The treatment-specific section is:

```yaml
reconstruction:
  coefficient: 1.0
  decoder_unit_norm: true
  auxk:
    mode: auxk
    coefficient: 0.03125
    width_ratio: 0.5
    dead_tokens_threshold: 10000000
```

The CPU smoke profile uses the same `K=4D`, `k_aux=D/2`, eight-cohort geometry
and a 128-token threshold so its second step exercises nonempty dead state.

## Reproduction checks

```bash
JAX_PLATFORMS=cpu .venv/bin/python \
  recipes/fuzzy_topk_reconstruction_auxk/train.py \
  --profile smoke --output-dir /tmp/fuzzy-topk-reconstruction-auxk-smoke

.venv/bin/pytest -q \
  tests/test_fuzzy_topk_reconstruction.py \
  tests/test_recipe_fuzzy_topk_reconstruction.py
```

The tests pin the custom VJP against a literal oracle, verify that AuxK cannot
differentiate through its residual target, check global activity counts and
dead-age state, and bill the default path at exactly `19MDH` per block.
