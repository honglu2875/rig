"""Algorithmic FLOP accounting by jaxpr traversal.

The training FLOP figure used by logs, artifacts, and equi-FLOP plots is
derived from the traced computation rather than a formula maintained by
hand. Changing width, depth, head count, or the shape of a block changes
the count automatically, because the count is read off the jaxpr that JAX
actually builds.

What this measures is *algorithmic* compute: the arithmetic the model
performs as mathematics, not the arithmetic the hardware issues. The two
differ, deliberately:

* Causal attention is billed for the full ``T x T`` square even though the
  flash kernel predicates off fully-masked tiles. Two runs of the same
  architecture must be comparable regardless of which kernel served them.
* An implementation that recomputes a value to save memory is billed once.
* Elementwise work is tracked but excluded from the headline, matching the
  ``6P`` convention that the reported figure is matmul work.

The consequence worth internalizing: *the FLOP count is a property of the
architecture, not of the backend*. Swapping ``dense`` attention for
``tpu_flash`` must not move the number, and a test asserts exactly that.

Extending it
------------
Primitives fall into four buckets. ``dot_general`` and
``conv_general_dilated`` are counted from their shapes. Structural ops
(reshape, transpose, gather) are free. Elementwise ops accumulate into a
separate total. Everything else -- an unrecognized primitive, or an opaque
kernel with no rule -- produces a warning rather than vanishing silently,
because a silent undercount is far worse than a loud one.

Two escape hatches exist for components whose real cost differs from their
traced cost:

``rules.with_kernel(name, rule)``
    Keyed on a ``pallas_call`` kernel name. Use when the arithmetic happens
    inside an opaque kernel that the tracer cannot see into.

``rules.with_scope(name, rule)``
    Keyed on a ``jit`` boundary name. Use when the arithmetic *is* visible
    but should not be counted at face value. Wrap the component in a named
    ``jax.jit`` and register a rule for that name; the walker applies the
    rule instead of descending.

The scope hatch is what a sparse mixture-of-experts needs. An MoE written
as "compute every expert, then mask to top-k" contains the full dense work
in its graph, and no graph analysis can know the mask discards it -- the
tracer sees real multiplications whose results are real. Counting it
correctly requires stating the intent:

    @partial(jax.jit, static_argnames=("experts", "top_k"))
    def moe_block(x, w, *, experts, top_k): ...

    def moe_rule(site):
        tokens, d_model = site.in_shapes[0][:2]
        _, _, d_ff = site.in_shapes[1]
        return 2 * site.params["top_k"] * tokens * d_model * d_ff * 3

    rules = default_rules().with_scope("moe_block", moe_rule)

See ``docs/FLOPS.md`` for the full checklist of when a new component needs
a rule.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import jax
from jax.extend import core


class FlopError(Exception):
    """Raised when a strict count meets work it cannot account for."""


# Ops that move or reinterpret bytes without arithmetic.
_STRUCTURAL = frozenset(
    {
        "reshape",
        "transpose",
        "broadcast_in_dim",
        "squeeze",
        "expand_dims",
        "slice",
        "dynamic_slice",
        "dynamic_update_slice",
        "concatenate",
        "convert_element_type",
        "copy",
        "device_put",
        "rev",
        "pad",
        "gather",
        "scatter",
        "scatter-add",
        "scatter_add",
        "iota",
        "split",
        "select_and_scatter",
        "bitcast_convert_type",
        "stop_gradient",
        "reduce_precision",
        "sharding_constraint",
        "squeeze_p",
        "tie_in",
        "empty",
        "opt_barrier",
        "optimization_barrier",
        "stack",
        "unstack",
        "select_and_gather_add",
        "clone",
    }
)

# Arithmetic that scales with array size rather than with a contraction.
_ELEMENTWISE = frozenset(
    {
        "add",
        "add_any",
        "sub",
        "mul",
        "div",
        "neg",
        "exp",
        "exp2",
        "expm1",
        "log",
        "log1p",
        "tanh",
        "logistic",
        "erf",
        "erfc",
        "erf_inv",
        "rsqrt",
        "sqrt",
        "cbrt",
        "pow",
        "integer_pow",
        "max",
        "min",
        "select_n",
        "and",
        "or",
        "not",
        "xor",
        "eq",
        "ne",
        "lt",
        # top_k lowers to a comparison variant on TPU; same class as "lt".
        "lt_to",
        "le",
        "gt",
        "ge",
        "sign",
        "abs",
        "floor",
        "ceil",
        "round",
        "rem",
        "nextafter",
        "clamp",
        "square",
        "logaddexp",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "sinh",
        "cosh",
        "asinh",
        "acosh",
        "atanh",
        "is_finite",
        "real",
        "imag",
        "conj",
        "complex",
        "shift_left",
        "shift_right_logical",
        "shift_right_arithmetic",
        "population_count",
        "clz",
        "reduce_sum",
        "reduce_max",
        "reduce_min",
        "reduce_prod",
        "reduce_and",
        "reduce_or",
        "reduce_xor",
        "argmax",
        "argmin",
        "cumsum",
        "cumlogsumexp",
        "cummax",
        "cummin",
        "cumprod",
        "sort",
        "top_k",
        "random_bits",
        "random_seed",
        "random_split",
        "random_wrap",
        "random_unwrap",
        "random_fold_in",
        "threefry2x32",
        "erf_inv_p",
        "rng_uniform",
        "rng_bit_generator",
        "nextafter_p",
    }
)

# Kernels the tracer cannot see inside. Without a rule these warn.
_OPAQUE = frozenset({"pallas_call", "tpu_custom_call", "custom_call", "ffi_call"})

# Carry sub-jaxprs that are entered exactly once.
_TRANSPARENT = frozenset(
    {
        "jit",
        "pjit",
        "closed_call",
        "core_call",
        "remat",
        "remat2",
        "checkpoint",
        "custom_vjp_call",
        "custom_vjp_call_jaxpr",
        "custom_jvp_call",
        "custom_jvp_call_jaxpr",
        "custom_lin",
        "custom_transpose_call",
        "named_call",
        "run_state",
        "linear_call",
    }
)


@dataclasses.dataclass(frozen=True)
class Site:
    """A rule's view of one equation, insulated from jaxpr internals."""

    name: str
    primitive: str
    in_shapes: tuple[tuple[int, ...], ...]
    out_shapes: tuple[tuple[int, ...], ...]
    params: Mapping[str, Any]

    def first_rank(self, rank: int) -> tuple[int, ...]:
        """Return the first input whose shape has ``rank`` dimensions."""

        for shape in self.in_shapes:
            if len(shape) == rank:
                return shape
        raise FlopError(
            f"{self.name!r}: expected an input of rank {rank}, saw "
            f"{[len(s) for s in self.in_shapes]}"
        )


Rule = Callable[[Site], int]


@dataclasses.dataclass(frozen=True)
class FlopRules:
    """Rules keyed by opaque-kernel name and by named ``jit`` boundary."""

    kernels: Mapping[str, Rule] = dataclasses.field(default_factory=dict)
    scopes: Mapping[str, Rule] = dataclasses.field(default_factory=dict)

    def with_kernel(self, name: str, rule: Rule) -> FlopRules:
        return dataclasses.replace(self, kernels={**self.kernels, name: rule})

    def with_scope(self, name: str, rule: Rule) -> FlopRules:
        return dataclasses.replace(self, scopes={**self.scopes, name: rule})


@dataclasses.dataclass(frozen=True)
class FlopBreakdown:
    """Result of a count. ``matmul`` is the headline figure."""

    matmul: int
    elementwise: int
    by_site: Mapping[str, int]
    warnings: tuple[str, ...]

    def per_token(self, tokens: int) -> int:
        if tokens <= 0:
            raise FlopError("tokens must be positive")
        return int(self.matmul // tokens)


class _Accumulator:
    def __init__(self) -> None:
        self.matmul = 0
        self.elementwise = 0
        self.by_site: dict[str, int] = {}
        self.warnings: list[str] = []

    def add(self, label: str, flops: int, scale: int = 1) -> None:
        value = int(flops) * scale
        self.matmul += value
        self.by_site[label] = self.by_site.get(label, 0) + value

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)


def _shape(var: Any) -> tuple[int, ...]:
    aval = getattr(var, "aval", None)
    return tuple(getattr(aval, "shape", ()) or ())


def _size(var: Any) -> int:
    return math.prod(_shape(var)) if _shape(var) else 1


def _site(eqn: Any) -> Site:
    return Site(
        name=str(eqn.params.get("name") or eqn.primitive.name),
        primitive=eqn.primitive.name,
        in_shapes=tuple(_shape(v) for v in eqn.invars),
        out_shapes=tuple(_shape(v) for v in eqn.outvars),
        params=eqn.params,
    )


def dot_general_flops(eqn: Any) -> int:
    """Two FLOPs (one multiply, one add) per element of the contraction."""

    lhs, rhs = _shape(eqn.invars[0]), _shape(eqn.invars[1])
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = eqn.params[
        "dimension_numbers"
    ]
    batch = math.prod(lhs[d] for d in lhs_batch) if lhs_batch else 1
    free_lhs = math.prod(
        lhs[d] for d in range(len(lhs)) if d not in lhs_contract and d not in lhs_batch
    )
    free_rhs = math.prod(
        rhs[d] for d in range(len(rhs)) if d not in rhs_contract and d not in rhs_batch
    )
    contracted = math.prod(lhs[d] for d in lhs_contract) if lhs_contract else 1
    return 2 * batch * free_lhs * free_rhs * contracted


def conv_flops(eqn: Any) -> int:
    out = _shape(eqn.outvars[0])
    kernel = _shape(eqn.invars[1])
    # Every output element accumulates over the kernel's input channels.
    return 2 * math.prod(out) * math.prod(kernel[2:]) * kernel[1]


def _sub_jaxprs(eqn: Any) -> list[Any]:
    found: list[Any] = []
    for value in eqn.params.values():
        candidates = value if isinstance(value, (list, tuple)) else [value]
        for item in candidates:
            if isinstance(item, core.ClosedJaxpr):
                found.append(item.jaxpr)
            elif isinstance(item, core.Jaxpr):
                found.append(item)
    return found


def _multiplicity(eqn: Any, accumulator: _Accumulator) -> int:
    """How many times the body of a higher-order primitive runs."""

    name = eqn.primitive.name
    if name == "scan":
        return int(eqn.params.get("length", 1))
    if name == "shard_map":
        # The body is written from one device's point of view. Scaling by the
        # mesh recovers the global figure for partitioned work; a fully
        # replicated region would be counted once per device instead, which
        # is why training wraps only sharded computation.
        mesh = eqn.params.get("mesh")
        return int(getattr(mesh, "size", 1) or 1)
    if name in ("while", "while_loop"):
        accumulator.warn(
            "while loop has a data-dependent trip count; its body is counted once"
        )
        return 1
    return 1


def _walk(jaxpr: Any, rules: FlopRules, accumulator: _Accumulator, scale: int) -> None:
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        site = _site(eqn)

        # An explicit rule always wins, and stops the walk at that boundary.
        rule = rules.scopes.get(site.name) if name in _TRANSPARENT else None
        if rule is None and name in _OPAQUE:
            rule = rules.kernels.get(site.name)
        if rule is not None:
            accumulator.add(site.name, rule(site), scale)
            continue

        if name == "dot_general":
            accumulator.add("dot_general", dot_general_flops(eqn), scale)
            continue
        if name == "conv_general_dilated":
            accumulator.add("conv", conv_flops(eqn), scale)
            continue
        if name in _OPAQUE:
            accumulator.warn(
                f"opaque kernel {site.name!r} has no rule: its arithmetic is NOT "
                f"counted. Register one with rules.with_kernel({site.name!r}, ...)"
            )
            continue

        if name == "cond":
            # Branches are exclusive; bill the most expensive one.
            branches = _sub_jaxprs(eqn)
            best = None
            for branch in branches:
                probe = _Accumulator()
                _walk(branch, rules, probe, 1)
                if best is None or probe.matmul > best.matmul:
                    best = probe
            if best is not None:
                accumulator.matmul += best.matmul * scale
                accumulator.elementwise += best.elementwise * scale
                for label, value in best.by_site.items():
                    accumulator.by_site[label] = (
                        accumulator.by_site.get(label, 0) + value * scale
                    )
                for message in best.warnings:
                    accumulator.warn(message)
            continue

        children = _sub_jaxprs(eqn)
        if children:
            inner = scale * _multiplicity(eqn, accumulator)
            for child in children:
                _walk(child, rules, accumulator, inner)
            continue

        if name in _STRUCTURAL:
            continue
        if name in _ELEMENTWISE:
            accumulator.elementwise += sum(_size(v) for v in eqn.outvars) * scale
            continue

        accumulator.warn(
            f"unrecognized primitive {name!r} is not counted; classify it in "
            f"rig.flops or give it a rule"
        )


def count_jaxpr(jaxpr: Any, rules: FlopRules | None = None) -> FlopBreakdown:
    """Count a jaxpr that has already been traced."""

    accumulator = _Accumulator()
    _walk(jaxpr, rules or default_rules(), accumulator, 1)
    return FlopBreakdown(
        matmul=accumulator.matmul,
        elementwise=accumulator.elementwise,
        by_site=dict(accumulator.by_site),
        warnings=tuple(accumulator.warnings),
    )


def count_flops(
    fn: Callable[..., Any],
    *args: Any,
    rules: FlopRules | None = None,
    strict: bool = False,
    **kwargs: Any,
) -> FlopBreakdown:
    """Trace ``fn`` and count it. Nothing is executed and nothing is allocated."""

    jaxpr = jax.make_jaxpr(fn)(*args, **kwargs).jaxpr
    breakdown = count_jaxpr(jaxpr, rules)
    if strict and breakdown.warnings:
        raise FlopError("; ".join(breakdown.warnings))
    return breakdown


def count_training_flops(
    loss_fn: Callable[..., Any],
    params: Any,
    *args: Any,
    rules: FlopRules | None = None,
    strict: bool = False,
    **kwargs: Any,
) -> FlopBreakdown:
    """Count one forward and backward pass of ``loss_fn`` over ``params``."""

    return count_flops(
        jax.grad(loss_fn), params, *args, rules=rules, strict=strict, **kwargs
    )


def _flash_attention_rule(site: Site) -> int:
    """Bill causal attention as the full square, per kernel invocation.

    The three kernels split the textbook cost evenly. Forward computes
    ``QK^T`` and ``AV``; the dq kernel computes ``dP = dO V^T`` and
    ``dQ = dS K``; the dkv kernel computes ``dV = P^T dO`` and
    ``dK = dS^T Q``. Each pair is ``4 B H T^2 D``, so a full step bills
    ``12 B H T^2 D`` -- the same figure dense attention traces to, which is
    the property that keeps the two backends comparable.
    """

    batch, heads, sequence, head_dim = site.first_rank(4)
    return 4 * batch * heads * sequence * sequence * head_dim


def _grouped_matmul_rule(site: Site) -> int:
    """Bill a megablox grouped matmul from its shapes.

    All three forms cost the same ``2 m k n``, where ``m`` is the number of
    routed rows. Verified against the traced operands:

    ===============  ==========================  ==================
    call             large operands              cost
    ===============  ==========================  ==================
    ``gmm``          ``[m, k]``, ``[E, k, n]``   ``2 m k n``
    ``gmm`` (dX)     ``[m, n]``, ``[E, k, n]``   ``2 m k n``
    ``tgmm`` (dW)    ``[m, k]``, ``[m, n]``      ``2 m k n``
    ===============  ==========================  ==================

    Every one of the ``m`` rows is multiplied by exactly one expert's
    ``[k, n]``, so grouping moves work between experts without creating or
    removing any. That is what makes a routed model's FLOP count exact and
    static despite ``group_sizes`` being data: ``m`` is ``tokens * top_k``, so
    the count bills the ``top_k`` experts each token actually visits. Billing
    the whole weight tensor would inflate a top-2-of-8 model by 4x and destroy
    the equi-FLOP comparison a sparse ladder exists to make.

    Smaller rank-0 and rank-1 operands are group metadata and carry no
    arithmetic.
    """

    two_d = [shape for shape in site.in_shapes if len(shape) == 2]
    three_d = [shape for shape in site.in_shapes if len(shape) == 3]
    if three_d and two_d:
        rows = two_d[0][0]
        _, k, n = three_d[0]
        return 2 * rows * k * n
    if len(two_d) >= 2 and two_d[0][0] == two_d[1][0]:
        rows, k = two_d[0]
        n = two_d[1][1]
        return 2 * rows * k * n
    raise FlopError(
        f"{site.name!r}: not a recognized grouped matmul; shapes {site.in_shapes}"
    )


_FLASH_KERNELS = (
    "tpu_flash_causal_attention_fwd",
    "tpu_flash_causal_attention_bwd_dq",
    "tpu_flash_causal_attention_bwd_dkv",
)


def default_rules() -> FlopRules:
    """Rules for the kernels this repository ships."""

    rules = FlopRules()
    for kernel in _FLASH_KERNELS:
        rules = rules.with_kernel(kernel, _flash_attention_rule)
    # megablox does not name its pallas_call, so it arrives under the bare
    # primitive name. Nothing else in this repository ships an unnamed one,
    # and the rule refuses shapes it does not recognize rather than guessing.
    rules = rules.with_kernel("pallas_call", _grouped_matmul_rule)
    return rules


def describe(breakdown: FlopBreakdown) -> Iterator[tuple[str, str]]:
    """Rows for a console table, largest contributor first."""

    ordered: Sequence[tuple[str, int]] = sorted(
        breakdown.by_site.items(), key=lambda item: -item[1]
    )
    for label, value in ordered:
        share = 100.0 * value / breakdown.matmul if breakdown.matmul else 0.0
        yield label, f"{value:,} ({share:.1f}%)"
