from __future__ import annotations

import hashlib
import io
import json
import math
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from rig.cohort import CohortError, build_cohort, validate_cohort
from rig.harness import (
    ConfigurationError,
    ResultValidationError,
    RunConfig,
    RecipeError,
    load_records,
    parse_result_line,
    rank_records,
    render_leaderboard,
    run_recipe,
    validate_result,
    verify_run,
)
from rig.harness.runner import _validate_payload_identity
from rig.plan import validate_recipe_plan


FAKE_TRAINER = r"""from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True)
parser.add_argument("--seed", required=True, type=int)
parser.add_argument("--profile", required=True)
parser.add_argument("--tag", action="append", default=[])
parser.add_argument("--seeded", action="store_true")
parser.add_argument("--stderr-message", default="")
parser.add_argument("--stderr-bytes", type=int, default=0)
parser.add_argument("--sleep-after-stderr", type=float, default=0.0)
parser.add_argument("--make-cache", action="store_true")
parser.add_argument("--evaluations-json")
parser.add_argument("--omit-checkpoint", action="store_true")
args = parser.parse_args()
if args.stderr_message:
    sys.stderr.write(args.stderr_message)
if args.stderr_bytes:
    sys.stderr.buffer.write(b"x" * args.stderr_bytes)
sys.stderr.flush()
if args.sleep_after_stderr:
    time.sleep(args.sleep_after_stderr)
output = Path(args.output_dir)
if args.make_cache:
    (output / ".jax_cache").mkdir()
    (output / ".jax_cache" / "compiled.bin").write_bytes(b"temporary")
if not args.omit_checkpoint:
    (output / "model.npz").write_bytes(b"tiny checkpoint")
(output / "training.csv").write_text("step,train_loss\n1,2.5\n")
(output / "seen.json").write_text(json.dumps({"seed": args.seed, "profile": args.profile, "tag": args.tag, "seeded": args.seeded}))
result = {
    "schema_version": 1,
    "status": "ok",
    "track": "open",
    "profile": args.profile,
    "seed": args.seed,
    "checkpoint": None if args.omit_checkpoint else "model.npz",
    "artifacts": {"training_curve": "training.csv"},
    "metrics": {
        "train_seconds": 0.125,
        "tokens_processed": 96,
        "validation_loss": 2.5,
        "validation_tokens": 64,
        "compile_seconds": 0.75,
        "diagnostics": {"gradient_scale": -2.0},
    },
    "contract": {
        "model_id": "tiny-gpt-v1",
        "dataset_id": "tiny-data-v1",
        "tokenizer_id": "byte-v1",
        "sequence_length": 8,
    },
    "implementation": {
        "attention_backend": "dense",
        "loss_backend": "dense",
    },
    "system": {"platform": "test", "devices": 1},
}
if args.evaluations_json is not None:
    result["evaluations"] = json.loads(args.evaluations_json)
print("human log output")
print("RIG_RESULT=" + json.dumps(result, separators=(",", ":")))
"""

FAKE_CONFIG = b"steps: 1\n"
FAKE_CONFIG_SHA256 = hashlib.sha256(FAKE_CONFIG).hexdigest()


class HarnessRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        recipe = self.root / "recipes" / "tiny"
        recipe.mkdir(parents=True)
        (recipe / "train.py").write_text(FAKE_TRAINER, encoding="utf-8")
        (recipe / "config.yaml").write_bytes(FAKE_CONFIG)
        (recipe / "dev.yaml").write_bytes(FAKE_CONFIG)
        (recipe / "smoke.yaml").write_bytes(FAKE_CONFIG)
        (self.root / "uv.lock").write_bytes(b"version = 1\n")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def config(self, **changes: object) -> RunConfig:
        values: dict[str, object] = {
            "repo_root": self.root,
            "recipe": "tiny",
            "runs_dir": Path("runs"),
            "records_path": Path("records/runs.jsonl"),
            "python_executable": sys.executable,
            "trainer_args": ("--tag", "one", "--tag", "two"),
        }
        values.update(changes)
        values.setdefault(
            "plan", self.plan(profile=str(values.get("profile", "default")))
        )
        return RunConfig(**values)  # type: ignore[arg-type]

    @staticmethod
    def plan(
        *, profile: str = "default", expected_tokens: int = 96
    ) -> dict[str, object]:
        return {
            "schema_version": 3,
            "config_schema_version": 4,
            "config_sha256": FAKE_CONFIG_SHA256,
            "profile": profile,
            "context_preset": "tiny",
            "document_masking": False,
            "tier": "tiny",
            "run_kind": "full",
            "parameterization": "test",
            "weight_decay_policy": "weights_and_embeddings_only_v2",
            "declared_parameters": expected_tokens,
            "batch_size": 1,
            "sequence_length": 1,
            "tokens_per_step": 1,
            "target_tokens_per_parameter": 1.0,
            "achieved_tokens_per_parameter": 1.0,
            "schedule_steps": expected_tokens,
            "stop_after_step": None,
            "planned_tokens": expected_tokens,
            "expected_tokens": expected_tokens,
            "validation_predictions": 64,
            "base_learning_rate": 0.001,
            "batch_ratio": 1.0,
            "ladder_data_multiplier": 1.0,
        }

    @staticmethod
    def fresh10_evaluations(
        domain_tokens: dict[str, int] | None = None,
    ) -> dict[str, object]:
        tokens = domain_tokens or {
            name: 8_192
            for name in (
                "science",
                "medicine",
                "software",
                "history",
                "fiction",
                "government",
                "legal",
                "economics",
                "climate",
                "education",
            )
        }
        losses = {name: 2.0 + index / 10 for index, name in enumerate(tokens)}
        macro_loss = math.fsum(losses.values()) / len(losses)
        return {
            "fineweb": {
                "loss": 2.5,
                "perplexity": math.exp(2.5),
                "scored_tokens": 64,
                "seconds": 0.25,
                "canonical": True,
            },
            "fresh10": {
                "domains": {
                    name: {
                        "loss": losses[name],
                        "perplexity": math.exp(losses[name]),
                        "scored_tokens": count,
                        "seconds": 0.01 + index / 100,
                    }
                    for index, (name, count) in enumerate(tokens.items())
                },
                "macro_loss": macro_loss,
                "macro_perplexity": math.exp(macro_loss),
                "scored_tokens": sum(tokens.values()),
                "seconds": math.fsum(
                    0.01 + index / 100 for index in range(len(tokens))
                ),
            },
        }

    def test_run_captures_validates_records_and_forwards_args(self) -> None:
        outcome = run_recipe(self.config(target_loss=2.6))

        self.assertEqual(
            json.loads((outcome.run_dir / "seen.json").read_text()),
            {
                "seed": 1337,
                "profile": "default",
                "tag": ["one", "two"],
                "seeded": False,
            },
        )
        self.assertTrue((outcome.run_dir / "stdout.log").is_file())
        self.assertTrue((outcome.run_dir / "stderr.log").is_file())
        self.assertTrue((outcome.run_dir / "result.json").is_file())
        records = load_records(outcome.record_path)
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertTrue(record["qualified"])
        self.assertGreater(record["timing"]["observed_wall_seconds"], 0)
        self.assertEqual(record["metrics"]["train_seconds"], 0.125)
        self.assertEqual(record["target_loss"], 2.6)
        # The recipe's declared loss is authoritative; nothing recomputes it.
        self.assertEqual(record["metrics"]["validation_loss"], 2.5)
        self.assertEqual(record["metrics"]["compile_seconds"], 0.75)
        self.assertEqual(record["metrics"]["diagnostics"]["gradient_scale"], -2.0)
        self.assertEqual(record["system"], {"platform": "test", "devices": 1})
        self.assertEqual(
            record["implementation"],
            {"attention_backend": "dense", "loss_backend": "dense"},
        )
        self.assertEqual(record["artifacts"]["training_curve"]["path"], "training.csv")
        self.assertEqual(len(record["artifacts"]["training_curve"]["sha256"]), 64)
        self.assertEqual(len(record["checkpoint"]["sha256"]), 64)

    def test_multi_host_run_builds_controller_owned_distributed_launch(self) -> None:
        def localize(**arguments: object) -> list[str]:
            return list(arguments["command"])  # type: ignore[arg-type]

        with (
            mock.patch(
                "rig.harness.runner.build_distributed_launch_command",
                side_effect=localize,
            ) as build,
            mock.patch(
                "rig.harness.runner.socket.gethostname", return_value="slice-w-0"
            ),
        ):
            outcome = run_recipe(
                self.config(
                    tpu_vm_count=4,
                    tpu_vm_hosts="slice-w-[0-3]",
                )
            )

        remote_environment = build.call_args.kwargs["environment"]
        self.assertEqual(remote_environment["RIG_DISTRIBUTED"], "1")
        self.assertEqual(remote_environment["RIG_PROCESS_COUNT"], "4")
        self.assertEqual(remote_environment["RIG_CONTROLLER_HOSTNAME"], "slice-w-0")
        self.assertEqual(
            remote_environment["JAX_COMPILATION_CACHE_DIR"],
            f"/tmp/rig-jax-cache-{outcome.run_id}",
        )
        self.assertEqual(outcome.record["trainer_command"], outcome.record["command"])

    def test_interrupted_multi_host_run_cleans_exact_remote_workers(self) -> None:
        with (
            mock.patch(
                "rig.harness.runner.build_distributed_launch_command",
                return_value=["pdsh", "synthetic"],
            ),
            mock.patch(
                "rig.harness.runner._run_process", side_effect=KeyboardInterrupt
            ),
            mock.patch(
                "rig.harness.runner.terminate_distributed_workers", return_value=True
            ) as terminate,
        ):
            with self.assertRaises(KeyboardInterrupt):
                run_recipe(
                    self.config(
                        tpu_vm_count=4,
                        tpu_vm_hosts="slice-w-[0-3]",
                    )
                )

        arguments = terminate.call_args.kwargs
        self.assertEqual(arguments["host_expression"], "slice-w-[0-3]")
        self.assertEqual(arguments["host_count"], 4)
        self.assertEqual(arguments["script"].name, "train.py")
        self.assertIn("runs", arguments["output_dir"].parts)

    def test_rejects_reserved_trainer_flags_but_not_other_arguments(self) -> None:
        for flag in ("--output-dir", "--seed", "--profile", "--omit-checkpoint"):
            for arguments in (
                (flag, "value"),
                (f"{flag}=value",),
                ("--", flag, "value"),
            ):
                with self.subTest(arguments=arguments):
                    with self.assertRaisesRegex(ConfigurationError, "reserved flag"):
                        run_recipe(self.config(trainer_args=arguments))

        outcome = run_recipe(self.config(trainer_args=("--seeded",)))
        seen = json.loads((outcome.run_dir / "seen.json").read_text())
        self.assertTrue(seen["seeded"])

    def test_result_identity_must_exactly_match_config(self) -> None:
        trainer = self.root / "recipes" / "tiny" / "train.py"
        original = trainer.read_text(encoding="utf-8")
        variants = {
            "profile": original.replace(
                '"profile": args.profile,', '"profile": "wrong",'
            ),
            "seed": original.replace('"seed": args.seed,', '"seed": True,'),
            "missing": original.replace('    "profile": args.profile,\n', ""),
        }
        for label, source in variants.items():
            with self.subTest(label=label):
                trainer.write_text(source, encoding="utf-8")
                with self.assertRaisesRegex(
                    ResultValidationError, "must exactly match"
                ):
                    run_recipe(self.config())
        trainer.write_text(original, encoding="utf-8")

    def test_official_identity_requires_exact_v4_8_system(self) -> None:
        payload = {
            "track": "open",
            "profile": "official",
            "seed": 1337,
            "system": {
                "platform": "tpu",
                "device_count": 4,
                "local_device_count": 4,
                "process_count": 1,
                "device_kinds": ["TPU v4"],
            },
        }
        config = self.config(profile="official")
        _validate_payload_identity(payload, config)
        payload["system"]["device_count"] = 8
        with self.assertRaisesRegex(ResultValidationError, "device_count"):
            _validate_payload_identity(payload, config)

    def test_official_identity_accepts_configured_multi_host_v4_system(self) -> None:
        payload = {
            "track": "open",
            "profile": "official",
            "seed": 1337,
            "system": {
                "platform": "tpu",
                "device_count": 16,
                "local_device_count": 4,
                "process_count": 4,
                "device_kinds": ["TPU v4"],
            },
        }
        config = self.config(
            profile="official",
            tpu_vm_count=4,
            tpu_vm_hosts="slice-w-[0-3]",
        )
        _validate_payload_identity(payload, config)
        payload["system"]["process_count"] = 3
        with self.assertRaisesRegex(ResultValidationError, "process_count"):
            _validate_payload_identity(payload, config)

    def test_fixed_validation_prefix_count_is_enforced(self) -> None:
        outcome = run_recipe(self.config(expected_validation_tokens=64))
        self.assertEqual(outcome.record["metrics"]["validation_tokens"], 64)
        with self.assertRaisesRegex(ResultValidationError, "fixed validation prefix"):
            run_recipe(self.config(expected_validation_tokens=65))

    def test_fresh10_is_optional_without_expectations(self) -> None:
        outcome = run_recipe(self.config())
        self.assertNotIn("evaluations", outcome.record)

    def test_fresh10_contract_is_validated_and_preserved_in_record(self) -> None:
        expected_tokens = {
            name: 8_192
            for name in (
                "science",
                "medicine",
                "software",
                "history",
                "fiction",
                "government",
                "legal",
                "economics",
                "climate",
                "education",
            )
        }
        evaluations = self.fresh10_evaluations(expected_tokens)
        outcome = run_recipe(
            self.config(
                target_loss=2.6,
                expected_validation_tokens=64,
                expected_downstream_tokens=expected_tokens,
                trainer_args=("--evaluations-json", json.dumps(evaluations)),
            )
        )

        self.assertTrue(outcome.record["qualified"])
        self.assertEqual(outcome.record["metrics"]["validation_loss"], 2.5)
        self.assertEqual(outcome.record["evaluations"], evaluations)
        self.assertEqual(
            load_records(outcome.record_path)[0]["evaluations"], evaluations
        )

        evaluations["fresh10"]["macro_loss"] = 99  # type: ignore[index]
        self.assertNotEqual(
            outcome.record["evaluations"]["fresh10"]["macro_loss"],  # type: ignore[index]
            99,
        )

    def test_fresh10_expectations_require_evaluations(self) -> None:
        expected = {f"domain-{index}": 8_192 for index in range(10)}
        with self.assertRaisesRegex(ResultValidationError, "evaluations are required"):
            run_recipe(self.config(expected_downstream_tokens=expected))

    def test_invalid_fresh10_config_fails_before_launch(self) -> None:
        for expected in (
            {f"domain-{index}": 8_192 for index in range(9)},
            {**{f"domain-{index}": 8_192 for index in range(9)}, "domain-9": 0},
        ):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(
                    ConfigurationError, "expected_downstream_tokens"
                ):
                    run_recipe(self.config(expected_downstream_tokens=expected))

    def test_provenance_hashes_inputs_and_copies_configured_values(self) -> None:
        trainer = self.root / "recipes" / "tiny" / "train.py"
        recipe_config = self.root / "recipes" / "tiny" / "config.yaml"
        configured = {"data": {"manifest": "fineweb-α", "shards": 9}}
        outcome = run_recipe(self.config(provenance=configured))
        provenance = outcome.record["provenance"]

        self.assertEqual(provenance["data"], configured["data"])
        self.assertIsNot(provenance["data"], configured["data"])
        self.assertEqual(provenance["train_py"]["bytes"], trainer.stat().st_size)
        self.assertEqual(
            provenance["train_py"]["sha256"],
            hashlib.sha256(trainer.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            provenance["config_yaml"],
            {
                "path": "recipes/tiny/config.yaml",
                "sha256": hashlib.sha256(recipe_config.read_bytes()).hexdigest(),
                "bytes": recipe_config.stat().st_size,
            },
        )
        self.assertEqual(
            provenance["uv_lock"]["sha256"],
            hashlib.sha256((self.root / "uv.lock").read_bytes()).hexdigest(),
        )
        self.assertEqual(provenance["shared_python"]["files"], 0)
        self.assertEqual(provenance["shared_python"]["bytes"], 0)
        self.assertEqual(
            provenance["shared_python"]["sha256"], hashlib.sha256().hexdigest()
        )

        configured["data"]["shards"] = 99
        self.assertEqual(provenance["data"]["shards"], 9)

        with self.assertRaisesRegex(ConfigurationError, "harness-owned"):
            run_recipe(self.config(provenance={"train_py": {"spoofed": True}}))
        with self.assertRaisesRegex(ConfigurationError, "harness-owned"):
            run_recipe(self.config(provenance={"shared_python": {"spoofed": True}}))
        with self.assertRaisesRegex(ConfigurationError, "harness-owned"):
            run_recipe(self.config(provenance={"config_yaml": {"spoofed": True}}))

    def test_recipe_config_must_be_a_regular_sibling_file(self) -> None:
        recipe = self.root / "recipes" / "tiny"
        recipe_config = recipe / "config.yaml"
        recipe_config.unlink()
        with self.assertRaisesRegex(ConfigurationError, "configuration file not found"):
            run_recipe(self.config())

        target = recipe / "elsewhere.yaml"
        target.write_text("steps: 1\n", encoding="utf-8")
        recipe_config.symlink_to(target.name)
        with self.assertRaisesRegex(ConfigurationError, "configuration file not found"):
            run_recipe(self.config())

    def test_shared_python_provenance_changes_with_dependency_bytes(self) -> None:
        shared = self.root / "rig" / "kernels"
        shared.mkdir(parents=True)
        dependency = shared / "attention.py"
        dependency.write_text("PLAN = 128\n", encoding="utf-8")
        first = run_recipe(self.config()).record["provenance"]["shared_python"]
        dependency.write_text("PLAN = 512\n", encoding="utf-8")
        second = run_recipe(self.config()).record["provenance"]["shared_python"]
        self.assertEqual(first["files"], 1)
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertNotEqual(
            first["entries"][0]["sha256"], second["entries"][0]["sha256"]
        )

    def test_rejects_nonfinite_declared_metrics_and_provenance(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "finite JSON"):
            run_recipe(self.config(provenance={"bad": float("nan")}))

        trainer = self.root / "recipes" / "tiny" / "train.py"
        source = trainer.read_text(encoding="utf-8").replace(
            '"compile_seconds": 0.75,', '"compile_seconds": float("inf"),'
        )
        trainer.write_text(source, encoding="utf-8")
        with self.assertRaisesRegex(ResultValidationError, "finite JSON"):
            run_recipe(self.config())

    def test_stderr_is_teed_live_and_captured_byte_for_byte(self) -> None:
        captured = io.StringIO()
        result: list[object] = []
        config = self.config(
            trainer_args=(
                "--stderr-message",
                "live marker",
                "--sleep-after-stderr",
                "0.35",
            )
        )

        with mock.patch("rig.harness.runner.sys.stderr", captured):
            thread = threading.Thread(target=lambda: result.append(run_recipe(config)))
            thread.start()
            deadline = time.monotonic() + 2.0
            while (
                "live marker" not in captured.getvalue() and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            self.assertIn("live marker", captured.getvalue())
            self.assertTrue(
                thread.is_alive(), "stderr was not visible until after process exit"
            )
            thread.join(3.0)

        self.assertFalse(thread.is_alive())
        outcome = result[0]
        self.assertEqual((outcome.run_dir / "stderr.log").read_bytes(), b"live marker")

    def test_large_stderr_does_not_deadlock_and_timeout_keeps_partial_log(self) -> None:
        captured = io.StringIO()
        with mock.patch("rig.harness.runner.sys.stderr", captured):
            outcome = run_recipe(
                self.config(trainer_args=("--stderr-bytes", str(512 * 1024)))
            )
        self.assertEqual((outcome.run_dir / "stderr.log").stat().st_size, 512 * 1024)

        timeout_config = self.config(
            trainer_args=(
                "--stderr-message",
                "before timeout",
                "--sleep-after-stderr",
                "10",
            ),
            timeout_seconds=0.1,
        )
        with mock.patch("rig.harness.runner.sys.stderr", io.StringIO()):
            started = time.monotonic()
            with self.assertRaisesRegex(RecipeError, "timed out"):
                run_recipe(timeout_config)
            self.assertLess(time.monotonic() - started, 2.0)
        latest_run = max(
            (self.root / "runs").iterdir(), key=lambda path: path.stat().st_mtime_ns
        )
        self.assertEqual((latest_run / "stderr.log").read_bytes(), b"before timeout")

    def test_discards_per_run_compilation_cache_and_rejects_bad_timeouts(self) -> None:
        outcome = run_recipe(self.config(trainer_args=("--make-cache",)))
        self.assertFalse((outcome.run_dir / ".jax_cache").exists())

        for value in (True, float("nan"), float("inf"), 10**10_000):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaisesRegex(ConfigurationError, "timeout_seconds"):
                    run_recipe(self.config(timeout_seconds=value))

    def test_resolved_plan_is_recorded_and_must_match_profile(self) -> None:
        outcome = run_recipe(self.config())
        self.assertEqual(outcome.record["run_kind"], "full")
        self.assertEqual(outcome.record["constraints"]["training_tokens"], 96)
        self.assertEqual(len(outcome.record["plan"]["sha256"]), 64)

        with self.assertRaisesRegex(ConfigurationError, "plan profile"):
            run_recipe(self.config(plan=self.plan(profile="different")))

    def test_none_policy_never_writes_a_checkpoint(self) -> None:
        outcome = run_recipe(self.config(checkpoint_policy="none", profile="dev"))
        self.assertIsNone(outcome.checkpoint_path)
        self.assertFalse((outcome.run_dir / "model.npz").exists())
        self.assertIsNone(outcome.record["checkpoint"])
        self.assertIn("--omit-checkpoint", outcome.record["trainer_command"])

    def test_qualifying_retention_removes_nonqualifying_checkpoint(self) -> None:
        outcome = run_recipe(self.config(target_loss=2.0))
        self.assertFalse(outcome.record["qualified"])
        self.assertIsNone(outcome.checkpoint_path)
        self.assertFalse(outcome.record["checkpoint"]["retained"])
        validated = verify_run(
            outcome.run_dir,
            require_checkpoint=False,
            allow_missing_checkpoint=True,
        )
        self.assertIsNone(validated.checkpoint_path)

    def test_research_run_can_explicitly_omit_checkpoint_and_be_reverified(
        self,
    ) -> None:
        outcome = run_recipe(
            self.config(
                profile="dev",
                checkpoint_policy="none",
            )
        )
        self.assertIsNone(outcome.checkpoint_path)
        self.assertIsNone(outcome.record["checkpoint"])
        self.assertEqual(
            outcome.record["provenance"]["config_yaml"]["path"],
            "recipes/tiny/dev.yaml",
        )
        validated = verify_run(
            outcome.run_dir,
            require_checkpoint=False,
        )
        self.assertIsNone(validated.checkpoint_path)
        with self.assertRaisesRegex(ResultValidationError, "checkpoint is required"):
            verify_run(outcome.run_dir)

    def test_plan_training_token_budget_is_enforced_and_recorded(self) -> None:
        outcome = run_recipe(self.config())
        self.assertEqual(outcome.record["constraints"]["training_tokens"], 96)
        with self.assertRaisesRegex(ResultValidationError, "training-token budget"):
            run_recipe(self.config(plan=self.plan(expected_tokens=95)))
        invalid = self.plan()
        invalid["expected_tokens"] = 0
        with self.assertRaisesRegex(ConfigurationError, "expected_tokens"):
            run_recipe(self.config(plan=invalid))

        wrong_config = self.plan()
        wrong_config["config_sha256"] = "c" * 64
        with self.assertRaisesRegex(ConfigurationError, "config_sha256"):
            run_recipe(self.config(plan=wrong_config))

    def test_cohort_must_align_with_the_plan_and_run_configuration(self) -> None:
        plan = validate_recipe_plan(self.plan())
        cohort = build_cohort(
            plan=plan,
            dataset_id="tiny-data-v1",
            tokenizer_id="byte-v1",
            dataset_provenance={
                "dataset": {
                    "manifest": {"canonical_sha256": "b" * 64},
                    "train_files": ["train.bin"],
                    "validation_files": ["val.bin"],
                    "validation_prefix_tokens": 64,
                }
            },
            accelerator="TPU v4",
            tpu_vm_count=1,
            chips_per_host=4,
            target_loss=3.28,
        )
        assert cohort is not None

        outcome = run_recipe(self.config(cohort=cohort))
        self.assertEqual(outcome.record["cohort_id"], cohort["cohort_id"])

        with self.assertRaisesRegex(ConfigurationError, "cohort profile"):
            run_recipe(
                self.config(
                    profile="other",
                    plan=self.plan(profile="other"),
                    cohort=cohort,
                )
            )

    def test_rejects_recipe_and_checkpoint_path_traversal(self) -> None:
        with self.assertRaises(ConfigurationError):
            run_recipe(self.config(recipe="../tiny"))

        run_dir = self.root / "manual"
        run_dir.mkdir()
        outside = self.root / "outside.npz"
        outside.write_bytes(b"outside")
        payload = {
            "schema_version": 1,
            "status": "ok",
            "checkpoint": "../outside.npz",
            "metrics": {
                "train_seconds": 1.0,
                "tokens_processed": 1,
                "validation_loss": 1.0,
            },
        }
        with self.assertRaisesRegex(ResultValidationError, "escapes"):
            validate_result(payload, run_dir=run_dir)


class ProtocolAndScoringTests(unittest.TestCase):
    def test_evaluation_schema_rejects_inconsistent_fresh10_aggregates(self) -> None:
        evaluations = HarnessRunTests.fresh10_evaluations()
        expected = {
            name: row["scored_tokens"]
            for name, row in evaluations["fresh10"]["domains"].items()  # type: ignore[index,union-attr]
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "model.npz").write_bytes(b"checkpoint")
            base_payload = {
                "schema_version": 1,
                "status": "ok",
                "checkpoint": "model.npz",
                "metrics": {
                    "train_seconds": 1.0,
                    "tokens_processed": 1,
                    "validation_loss": 2.5,
                    "validation_tokens": 64,
                },
                "evaluations": evaluations,
            }
            validated = validate_result(
                base_payload,
                run_dir=run_dir,
                expected_validation_tokens=64,
                expected_downstream_tokens=expected,
            )
            self.assertEqual(validated.evaluations, evaluations)

            mutations = {
                "fineweb canonical": lambda value: value["evaluations"][
                    "fineweb"
                ].update(canonical=False),
                "fineweb loss": lambda value: value["evaluations"]["fineweb"].update(
                    loss=2.4
                ),
                "macro loss": lambda value: value["evaluations"]["fresh10"].update(
                    macro_loss=1.0
                ),
                "macro perplexity": lambda value: value["evaluations"][
                    "fresh10"
                ].update(macro_perplexity=1.0),
                "total tokens": lambda value: value["evaluations"]["fresh10"].update(
                    scored_tokens=81_919
                ),
                "domain tokens": lambda value: next(
                    iter(value["evaluations"]["fresh10"]["domains"].values())
                ).update(scored_tokens=8_191),
                "nonfinite seconds": lambda value: next(
                    iter(value["evaluations"]["fresh10"]["domains"].values())
                ).update(seconds=float("inf")),
                "nonpositive perplexity": lambda value: next(
                    iter(value["evaluations"]["fresh10"]["domains"].values())
                ).update(perplexity=0.0),
                "inconsistent perplexity": lambda value: next(
                    iter(value["evaluations"]["fresh10"]["domains"].values())
                ).update(perplexity=42.0),
                "inconsistent total seconds": lambda value: value["evaluations"][
                    "fresh10"
                ].update(seconds=42.0),
            }
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    candidate = json.loads(json.dumps(base_payload))
                    mutate(candidate)
                    with self.assertRaises(ResultValidationError):
                        validate_result(
                            candidate,
                            run_dir=run_dir,
                            expected_validation_tokens=64,
                            expected_downstream_tokens=expected,
                        )

    def test_fresh10_domain_names_must_match_expected_mapping(self) -> None:
        evaluations = HarnessRunTests.fresh10_evaluations()
        expected = {
            name: row["scored_tokens"]
            for name, row in evaluations["fresh10"]["domains"].items()  # type: ignore[index,union-attr]
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "model.npz").write_bytes(b"checkpoint")
            domains = evaluations["fresh10"]["domains"]  # type: ignore[index]
            domains["unexpected"] = domains.pop("science")  # type: ignore[union-attr]
            payload = {
                "schema_version": 1,
                "status": "ok",
                "checkpoint": "model.npz",
                "metrics": {
                    "train_seconds": 1.0,
                    "tokens_processed": 1,
                    "validation_loss": 2.5,
                },
                "evaluations": evaluations,
            }
            with self.assertRaisesRegex(ResultValidationError, "domain names"):
                validate_result(
                    payload,
                    run_dir=run_dir,
                    expected_downstream_tokens=expected,
                )

    def test_empty_leaderboard_renders(self) -> None:
        rendered = render_leaderboard([])
        self.assertIn("No qualifying runs.", rendered)

    def test_result_must_be_final_line_and_finite(self) -> None:
        with self.assertRaises(ResultValidationError):
            parse_result_line('RIG_RESULT={"schema_version":1}\nlate log\n')
        with self.assertRaises(ResultValidationError):
            parse_result_line("RIG_RESULT={not json}\n")

    def test_optional_implementation_provenance_must_be_an_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "model.npz").write_bytes(b"checkpoint")
            payload = {
                "schema_version": 1,
                "status": "ok",
                "checkpoint": "model.npz",
                "implementation": ["not", "an", "object"],
                "metrics": {
                    "train_seconds": 1.0,
                    "tokens_processed": 1,
                    "validation_loss": 1.0,
                },
            }
            with self.assertRaisesRegex(ResultValidationError, "implementation"):
                validate_result(payload, run_dir=run_dir)

    @staticmethod
    def cohort(manifest_sha256: str = "b" * 64) -> dict[str, object]:
        plan = validate_recipe_plan(HarnessRunTests.plan(profile="tiny"))
        cohort = build_cohort(
            plan=plan,
            dataset_id="tiny-data-v1",
            tokenizer_id="byte-v1",
            dataset_provenance={
                "dataset": {
                    "manifest": {"canonical_sha256": manifest_sha256},
                    "train_files": ["train.bin"],
                    "validation_files": ["val.bin"],
                    "validation_prefix_tokens": 64,
                }
            },
            accelerator="TPU v4",
            tpu_vm_count=1,
            chips_per_host=4,
            target_loss=3.28,
        )
        assert cohort is not None
        return cohort

    @staticmethod
    def ranking_record(
        name: str,
        seconds: float,
        cohort: dict[str, object],
        *,
        qualified: bool = True,
        run_kind: str = "full",
    ) -> dict[str, object]:
        return {
            "run_id": name + "-run",
            "recipe": name,
            "status": "ok",
            "qualified": qualified,
            "profile": "tiny",
            "run_kind": run_kind,
            "cohort": cohort,
            "cohort_id": cohort["cohort_id"],
            "timing": {"observed_wall_seconds": 100.0 / seconds},
            "metrics": {
                "train_seconds": seconds,
                "tokens_processed": 96,
                "validation_loss": 2.0,
            },
        }

    def test_cohort_ranking_uses_synchronized_time_and_excludes_diagnostics(
        self,
    ) -> None:
        cohort = self.cohort()
        records = [
            self.ranking_record("dense", 2.0, cohort),
            self.ranking_record("moe", 1.0, cohort),
            self.ranking_record("diagnostic", 0.5, cohort, run_kind="diagnostic"),
            self.ranking_record("unqualified", 0.25, cohort, qualified=False),
        ]
        ranked = rank_records(
            records, cohort_id=str(cohort["cohort_id"]), profile="tiny"
        )
        self.assertEqual([item["recipe"] for item in ranked], ["moe", "dense"])
        rendered = render_leaderboard(ranked, cohort_id=str(cohort["cohort_id"]))
        self.assertIn(str(cohort["cohort_id"])[:12], rendered)
        self.assertIn("moe", rendered)

    def test_ranker_refuses_to_mix_cohorts(self) -> None:
        first = self.cohort("b" * 64)
        second = self.cohort("c" * 64)
        records = [
            self.ranking_record("first", 1.0, first),
            self.ranking_record("second", 2.0, second),
        ]
        with self.assertRaisesRegex(ValueError, "multiple cohorts"):
            rank_records(records, profile="tiny")
        selected = rank_records(
            records, cohort_id=str(second["cohort_id"]), profile="tiny"
        )
        self.assertEqual([item["recipe"] for item in selected], ["second"])

    def test_cohort_validation_is_strict_even_when_rehashed(self) -> None:
        cohort = self.cohort()
        body = {key: value for key, value in cohort.items() if key != "cohort_id"}
        body["surprise"] = True
        cohort = {
            **body,
            "cohort_id": hashlib.sha256(
                json.dumps(
                    body,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
        }

        with self.assertRaisesRegex(CohortError, "unknown field"):
            validate_cohort(cohort)
        self.assertEqual(rank_records([self.ranking_record("bad", 1.0, cohort)]), [])


if __name__ == "__main__":
    unittest.main()
