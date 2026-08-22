---
name: seal-study
description: "Turn completed rig runs into an auditable study: verify provenance, export the standard Hugging Face log package, write its README, publish compact and full browser payloads, add a concise GitHub report, update catalogs, verify uploads, and commit the intended files. Use when asked to seal, archive, upload, report, or publish experiment results."
---

# Seal a study

The Hugging Face dataset `quintic/rig-logs` is the archive of record. It holds
raw logs, exact provenance/reproduction information, `snapshot.json.gz`, and
`full.json.gz`. Findings HTML belongs in `docs/reports/` in the GitHub repo,
not in Hugging Face. Keep one-off analysis/report builders in `/tmp`, not a
tracked `studies/` folder.

## Audit before interpreting

1. Identify every intended run in `runs/`; exclude failed, stopped, partial,
   or superseded artifacts explicitly. Require `result.json.status == "ok"`
   and run `.venv/bin/rig verify <run>`.
2. Extract exact validation values, steps, tokens, throughput, dataset and
   validation identities, topology/process/device count, seed, resolved config,
   source commit, trainer/shared hashes, and dirty status. Never rewrite
   recorded provenance. Explain harmless dirt (for example an untracked editor
   swap) and why it cannot affect training; otherwise treat it as a real
   comparability problem.
3. Check the scientific coordinates programmatically. If a baseline comes from
   an earlier HF study, download its exact raw run and ledger entry so the new
   study is self-contained. Do not substitute a similarly named run.
4. Use permutation-invariant expert summaries across seeds/treatments. Expert
   ordinal `j` has no cross-run identity; busiest/idlest ratios, entropy,
   dispersion, sorted loads, and within-layer correlations are valid.
5. State the evidence boundary. A matched one-seed mechanism test may reject a
   bad rule, but does not estimate a stable effect size.

## Build the standard archive

Stage only the selected complete runs, then use the maintained exporter:

```bash
.venv/bin/rig report --runs /tmp/<study>-runs \
  --study-export-target /tmp/<study>-export \
  --study-name <study> --select '<anchored-regex>'
```

The result is `/tmp/<study>-export/<study>/` containing:

```text
README.md
records.jsonl
snapshot.json.gz
full.json.gz
<canonical-run-name>/
  training.riglog
  diagnostics.riglog
  result.json
  metrics.json
  validation.csv
```

The snapshot is a fast curves-first view and should remain roughly ≤1 MB when
practical; it may contain more than loss. `full.json.gz` contains every chart
point for the explicit “Load full report” action. Raw `.riglog` remains
authoritative. Verify exported run count, unique canonical folder names,
ledger count, required files, log readability, and both gzip JSON payloads.
Scientific coordinates must be represented in archive names; a collision must
fail rather than overwrite. `rig/report.py::_study_run_name` is the naming
authority.

Replace the exporter's intentionally empty README with a concise study card:

- exact setup/grid, seed(s), context, batch, LR, TPP/tokens/steps, optimizer,
  data and fixed validation, hardware;
- baseline origin and exact source/config/trainer/shared hashes;
- provenance exceptions without sanitizing the ledger;
- an exact result table, metric definitions, interpretation, and limitations;
- commit-pinned reproduction commands, including recipe options after `--`;
- file/layout notes, a short `huggingface_hub` + `rig.logpack` loading example,
  and links to the GitHub report and related HF study.

Examples: `moe-no-bias`, `moe-router-aux-125M`, and
`moe-expert-load-scaling-125M` in
`https://huggingface.co/datasets/quintic/rig-logs/tree/main/`.

## Add the GitHub findings report

For a focused ablation, prefer a small static page such as
`docs/reports/moe-ablations.html` or
`docs/reports/expert-load-scaling.html`: inline SVG, system fonts, no runtime
fetch, exact endpoints, and only reductions relevant to the claim. Label
historical SD as scale context rather than a confidence interval. Put detailed
raw-series exploration in `docs/reports/study-browser.html`.

Update all three catalogs:

1. `docs/reports/README.md`: total study/run counts, report-size policy,
   hardware note, contents row, findings, limits, and reproduction command.
2. `docs/reports/study-browser.html`: decode its embedded `report-data`, append
   `{name,title,tier,runs,snapshot,full}`, then regenerate with
   `rig.report.build_study_browser`; do not hand-edit generated HTML.
3. The HF root `README.md`: counts, topology table, report policy/link, contents
   row, and findings section. Preserve its YAML frontmatter and existing
   catalog.

## Publish and prove it

Assemble a temporary upload root containing only the updated root `README.md`
and the new `<study>/` folder. After the user authorizes the external write:

```bash
set -a
source .env.hf
set +a
hf upload quintic/rig-logs /tmp/<upload-root> . --repo-type dataset \
  --commit-message "Add <study>"
```

Never display `.env.hf` or `HF_TOKEN`. Capture the immutable HF commit. Download
the root card and complete study from that revision into a fresh `/tmp`
directory, then compare relative file sets, byte counts, and SHA-256 hashes
against the upload root. An upload message alone is not verification.

Run `make check`, inspect `git diff --check`, and validate HTML parsing plus
the browser's embedded study count. Explicitly `git add` only intended source,
tests, `docs/reports/`, and `docs/skills/` files—never `runs/`, `.env.hf`,
temporary exports, or editor swaps. Commit and push the requested branch only
when authorized. Keep raw local runs until the HF verification succeeds;
delete or clean them only on an explicit request.
