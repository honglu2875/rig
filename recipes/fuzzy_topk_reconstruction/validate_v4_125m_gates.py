#!/usr/bin/env python3
"""Fail closed before the full v4-32 reconstruction treatment runs."""

from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path


RUNS = Path(os.environ["RIG_RECON_RUNS_ROOT"])
PARENT_TOKENS_PER_SECOND = 578_142.0


def one(pattern: str) -> dict:
    matches = sorted(glob.glob(str(RUNS / pattern / "result.json")))
    if len(matches) != 1:
        raise SystemExit(f"expected one gate result for {pattern!r}; got {matches}")
    with Path(matches[0]).open(encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("status") != "ok":
        raise SystemExit(f"gate is not complete and valid: {matches[0]}")
    return result


reconstruction = one("*-125m-v4-reconstruction-gate-v2-s1350-*")
auxk = one("*-125m-v4-reconstruction-auxk-gate-v2-s135-*")

for label, result in (("reconstruction", reconstruction), ("auxk", auxk)):
    metrics = result["metrics"]
    required = (
        float(metrics["train_loss"]),
        float(metrics["validation_loss"]),
        float(metrics["tokens_per_second"]),
        float(metrics["training_objective"]),
        float(metrics["reconstruction"]["model"]["fuzzy_reconstruction.nmse"]),
    )
    if not all(math.isfinite(value) for value in required):
        raise SystemExit(f"{label} gate contains a non-finite metric: {required}")
    if int(metrics["training_steps"]) != 120:
        raise SystemExit(f"{label} gate did not complete exactly 120 steps")
    if int(metrics["sparsity_diagnostic_point_count"]) < 3:
        raise SystemExit(f"{label} gate lacks its three full-feature captures")

reconstruction_rate = float(reconstruction["metrics"]["tokens_per_second"])
if reconstruction_rate < 0.5 * PARENT_TOKENS_PER_SECOND:
    raise SystemExit(
        "reconstruction throughput is below the predeclared half-parent floor: "
        f"{reconstruction_rate:.1f}"
    )

reconstruction_seconds = float(reconstruction["metrics"]["train_seconds"])
auxk_seconds = float(auxk["metrics"]["train_seconds"])
if auxk_seconds > 1.15 * reconstruction_seconds:
    raise SystemExit(
        "AuxK exceeds the predeclared 15% incremental time ceiling: "
        f"{auxk_seconds / reconstruction_seconds - 1.0:.3%}"
    )

aux_metrics = auxk["metrics"]["reconstruction"]["model"]
for metric in (
    "fuzzy_reconstruction.auxk_nmse",
    "fuzzy_reconstruction.auxk_positive_fraction",
):
    if not math.isfinite(float(aux_metrics[metric])):
        raise SystemExit(f"AuxK gate contains non-finite {metric}")

print(
    json.dumps(
        {
            "status": "admitted",
            "reconstruction_tokens_per_second": reconstruction_rate,
            "auxk_incremental_time_fraction": (
                auxk_seconds / reconstruction_seconds - 1.0
            ),
        },
        sort_keys=True,
    )
)
