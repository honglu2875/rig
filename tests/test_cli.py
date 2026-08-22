from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rig import cli
from rig.config import ConfigError, LocalConfig
from rig.data import DataError, Fresh10Domain, PreparedDataset, PreparedFresh10
from rig.doctor import check_prepared_data
from rig.plan import RecipePlan


class CliTests(unittest.TestCase):
    def test_requested_data_diagnostics_fail_when_cache_is_missing(self) -> None:
        with patch(
            "rig.doctor.verify_dataset",
            side_effect=DataError("missing dataset shard"),
        ):
            result = check_prepared_data(Path("/dev/shm"), "official")
        self.assertEqual(result.status, "error")
        self.assertIn("make prepare", result.hint or "")

    def test_run_surface_exposes_only_supported_research_overrides(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(
            [
                "run",
                "reference",
                "--tier",
                "500m",
                "--context",
                "8k",
                "--tokens-per-parameter",
                "5",
                "--base-learning-rate",
                "0.001",
                "--batch-size",
                "128",
                "--stop-after-step",
                "100",
            ]
        )
        self.assertEqual(args.tier, "500m")
        self.assertEqual(args.context, "8k")
        self.assertEqual(args.tokens_per_parameter, 5.0)
        self.assertEqual(args.batch_size, 128)
        for removed in ("--steps", "--train-tokens", "--track", "--layers"):
            with self.subTest(removed=removed), self.assertRaises(SystemExit):
                parser.parse_args(["run", "reference", removed, "1"])

    def test_run_and_profile_accept_recipe_arguments_only_after_boundary(self) -> None:
        parser = cli.build_parser()
        invocations = (
            ["run", "variant", "--profile", "dev"],
            ["profile", "variant", "--output-dir", "profiles/test"],
        )
        for public in invocations:
            with self.subTest(command=public[0]):
                args = cli._parse_arguments(
                    parser,
                    [
                        *public,
                        "--",
                        "--variant-mode",
                        "fast",
                        "--variant-strength=0.5",
                    ],
                )
                self.assertEqual(
                    args.recipe_args,
                    ("--variant-mode", "fast", "--variant-strength=0.5"),
                )
                self.assertEqual(
                    cli._recipe_specific_trainer_args(args), list(args.recipe_args)
                )

        with self.assertRaises(SystemExit):
            parser.parse_args(["run", "variant", "--variant-mode", "fast"])
        with self.assertRaises(SystemExit):
            cli._parse_arguments(parser, ["run", "variant", "--"])
        for flag in ("--seed", "--tier", "--train-data", "--xprof-dir"):
            with self.subTest(flag=flag), self.assertRaisesRegex(
                ConfigError, f"harness-managed {flag}"
            ):
                cli._recipe_specific_trainer_args(
                    SimpleNamespace(recipe_args=(f"{flag}=value",))
                )

    def test_recipe_help_uses_the_recipe_parser_without_preparing_a_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = root / "recipes" / "variant"
            recipe.mkdir(parents=True)
            trainer = recipe / "train.py"
            trainer.write_text("pass\n", encoding="utf-8")
            (recipe / "dev.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            completed = SimpleNamespace(returncode=0)
            with (
                patch("rig.cli.repo_root", return_value=root),
                patch("rig.cli.load_config") as load_config,
                patch("rig.cli.subprocess.run", return_value=completed) as run,
            ):
                self.assertEqual(
                    cli.main(["run", "variant", "--", "--help"]), 0
                )

            load_config.assert_not_called()
            self.assertEqual(run.call_args.args[0][-2:], [str(trainer), "--help"])
            self.assertEqual(run.call_args.kwargs["cwd"], recipe)

    def test_run_passes_profile_as_harness_state_not_trainer_passthrough(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = root / "recipes" / "variant"
            recipe.mkdir(parents=True)
            (recipe / "train.py").write_text("pass\n", encoding="utf-8")
            (recipe / "config.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (recipe / "dev.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            prepared = PreparedDataset(
                name="tiny",
                root=root / "data",
                manifest_path=manifest,
                manifest_sha256="a" * 64,
                train_files=(root / "data" / "train.bin",),
                validation_files=(root / "data" / "val.bin",),
                train_tokens=1_000,
                validation_tokens=20,
                validation_prefix_tokens=20,
            )
            plan = RecipePlan(
                payload={
                    "tier": "500m",
                    "schedule_steps": 10,
                    "expected_tokens": 1_000,
                    "run_kind": "full",
                },
                sha256="b" * 64,
            )
            config = LocalConfig(
                data_path="data",
                artifacts_path="runs",
                default_profile="dev",
                checkpoint_policy="none",
                color="never",
            )
            args = cli._parse_arguments(
                cli.build_parser(),
                [
                    "run",
                    "variant",
                    "--profile",
                    "dev",
                    "--tier",
                    "500m",
                    "--batch-size",
                    "128",
                    "--name",
                    "profile-boundary",
                    "--skip-data-check",
                    "--",
                    "--variant-mode",
                    "fast",
                ]
            )
            outcome = SimpleNamespace(
                run_id="variant-profile-boundary-test",
                record={
                    "qualified": False,
                    "metrics": {
                        "validation_loss": 4.0,
                        "train_seconds": 1.0,
                        "tokens_processed": 1_000,
                    },
                },
            )
            with (
                patch("rig.cli.repo_root", return_value=root),
                patch("rig.cli.load_config", return_value=config),
                patch("rig.cli.resolve_recipe_plan", return_value=plan) as resolve,
                patch("rig.cli._require_prepared_dataset"),
                patch("rig.cli.resolve_preparation_manifest", return_value=manifest),
                patch("rig.cli.verify_dataset", return_value=prepared),
                patch("rig.cli.build_cohort", return_value=None),
                patch("rig.cli.run_recipe", return_value=outcome) as run,
            ):
                self.assertEqual(cli.command_run(args), 0)

            plan_arguments = resolve.call_args.kwargs["arguments"]
            self.assertEqual(plan_arguments.count("--profile"), 1)
            self.assertEqual(
                plan_arguments[plan_arguments.index("--profile") + 1], "dev"
            )
            run_config = run.call_args.args[0]
            self.assertEqual(run_config.profile, "dev")
            self.assertNotIn("--profile", run_config.trainer_args)
            self.assertIn("--tier", run_config.trainer_args)
            self.assertIn("--batch-size", run_config.trainer_args)
            self.assertIn("--variant-mode", plan_arguments)
            self.assertIn("--variant-mode", run_config.trainer_args)
            self.assertEqual(
                run_config.trainer_args[
                    run_config.trainer_args.index("--train-data") + 1
                ],
                str(prepared.train_files[0]),
            )
            self.assertEqual(
                run_config.trainer_args[
                    run_config.trainer_args.index("--val-data") + 1
                ],
                str(prepared.validation_files[0]),
            )

    def test_clone_copies_recipe_config_byte_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "recipes" / "source"
            source.mkdir(parents=True)
            (source / "train.py").write_text("print('train')\n", encoding="utf-8")
            config_bytes = b"steps: 20\r\nlearning_rate: 3.0e-4\r\n"
            (source / "config.yaml").write_bytes(config_bytes)
            (source / "dev.yaml").write_bytes(b"steps: 2\r\n")
            (source / "smoke.yaml").write_bytes(b"steps: 1\r\n")
            (source / "README.md").write_text("# Source\n", encoding="utf-8")
            args = cli.build_parser().parse_args(["clone", "source", "variant"])

            with patch("rig.cli.repo_root", return_value=root):
                self.assertEqual(cli.command_clone(args), 0)

            destination = root / "recipes" / "variant"
            self.assertEqual((destination / "config.yaml").read_bytes(), config_bytes)
            self.assertEqual((destination / "dev.yaml").read_bytes(), b"steps: 2\r\n")
            self.assertEqual(
                (destination / "smoke.yaml").read_bytes(), b"steps: 1\r\n"
            )
            self.assertEqual(
                (destination / "train.py").read_text(encoding="utf-8"),
                "print('train')\n",
            )
            self.assertTrue((destination / "README.md").is_file())

    def test_clone_requires_config_before_creating_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "recipes" / "source"
            source.mkdir(parents=True)
            (source / "train.py").write_text("print('train')\n", encoding="utf-8")
            args = cli.build_parser().parse_args(["clone", "source", "variant"])

            with patch("rig.cli.repo_root", return_value=root):
                with self.assertRaisesRegex(
                    ConfigError, "configuration does not exist"
                ):
                    cli.command_clone(args)

            self.assertFalse((root / "recipes" / "variant").exists())

    def test_non_run_unknown_arguments_are_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            cli.main(["doctor", "--not-a-doctor-flag"])

    def test_official_target_is_versioned_and_can_only_be_tightened(self) -> None:
        self.assertEqual(
            cli._effective_target_loss(
                "official", requested=None, development_default=99.0
            ),
            3.28,
        )
        self.assertEqual(
            cli._effective_target_loss(
                "official", requested=3.26, development_default=99.0
            ),
            3.26,
        )
        with self.assertRaisesRegex(ConfigError, "may not be easier"):
            cli._effective_target_loss(
                "official", requested=3.29, development_default=3.28
            )
        self.assertEqual(
            cli._effective_target_loss("dev", requested=None, development_default=4.0),
            4.0,
        )

    def test_official_validation_mismatch_fails_before_launch(self) -> None:
        prepared = PreparedDataset(
            name="tiny",
            root=Path("data"),
            manifest_path=Path("manifest.json"),
            manifest_sha256="a" * 64,
            train_files=(Path("train.bin"),),
            validation_files=(Path("val.bin"),),
            train_tokens=100,
            validation_tokens=100_000_000,
            validation_prefix_tokens=10_485_760,
        )
        matching = RecipePlan(
            payload={"validation_predictions": 10_485_760}, sha256="b" * 64
        )
        self.assertEqual(
            cli._validation_contract(matching, prepared, "official"),
            10_485_760,
        )
        mismatch = RecipePlan(
            payload={"validation_predictions": 100_000_000}, sha256="c" * 64
        )
        with self.assertRaisesRegex(ConfigError, "does not match"):
            cli._validation_contract(mismatch, prepared, "official")
        self.assertIsNone(cli._validation_contract(mismatch, prepared, "dev"))

    def test_wizard_accepts_defaults_and_returns_complete_config(self) -> None:
        defaults = LocalConfig()
        # Empty answers accept every current machine, run, and dataset default;
        # dataset preparation remains automatic.
        with patch("builtins.input", side_effect=[""] * 13) as prompt:
            (
                result,
                preparation_type,
                diagnostics,
                require_tpu,
                download,
                save,
            ) = cli._prepare_wizard(
                defaults,
                preparation_type="official",
                run_diagnostics=True,
                require_tpu=True,
                download=True,
                save=True,
            )
        self.assertEqual(result.dataset, defaults.dataset)
        self.assertEqual(result.train_shards, 9)
        self.assertEqual(result.default_profile, defaults.default_profile)
        self.assertEqual(preparation_type, "official")
        self.assertTrue(diagnostics)
        self.assertTrue(require_tpu)
        self.assertTrue(download)
        self.assertTrue(save)
        self.assertEqual(prompt.call_count, 13)

    def test_wizard_resets_the_shard_default_when_dataset_changes(self) -> None:
        defaults = LocalConfig(dataset="8B", train_shards=79)
        answers = ["", "", "2B", "", *("" for _ in range(9))]
        with patch("builtins.input", side_effect=answers):
            result, *_ = cli._prepare_wizard(
                defaults,
                preparation_type="official",
                run_diagnostics=True,
                require_tpu=True,
                download=True,
                save=True,
            )
        self.assertEqual(result.dataset, "2B")
        self.assertEqual(result.train_shards, 19)

    def test_noninteractive_dataset_change_does_not_reuse_the_old_prefix(self) -> None:
        args = cli.build_parser().parse_args(
            [
                "prepare",
                "--non-interactive",
                "--dataset",
                "2B",
                "--no-download",
                "--no-doctor",
                "--no-save",
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = LocalConfig(
                data_path="data",
                artifacts_path="runs",
                dataset="8B",
                train_shards=79,
            )
            with (
                patch("rig.cli.repo_root", return_value=root),
                patch("rig.cli.load_config", return_value=current),
                patch(
                    "rig.cli._route_for_config", wraps=cli._route_for_config
                ) as routed,
            ):
                self.assertEqual(cli.command_prepare(args), 0)

        proposed = routed.call_args.args[0]
        self.assertEqual(proposed.dataset, "2B")
        self.assertEqual(proposed.train_shards, 0)
        self.assertEqual(cli._route_for_config(proposed, "official").train_shards, 19)

    def test_prepare_dataset_and_shard_prefix_are_explicit(self) -> None:
        parser = cli.build_parser()
        prepared = parser.parse_args(
            ["prepare", "--dataset", "2B", "--train-shards", "10"]
        )
        self.assertEqual(prepared.dataset, "2B")
        self.assertEqual(prepared.train_shards, 10)
        with self.assertRaises(SystemExit):
            parser.parse_args(["prepare", "--training-tokens", "1250000000"])
        action = next(
            item
            for item in parser._subparsers._group_actions[0].choices["prepare"]._actions
            if item.dest == "dataset"
        )
        self.assertIn("immutable corpus", action.help)

    def test_prepare_routes_scaled_data_to_dedicated_subfolder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "cache"
            scaled = base / "fineweb-scaled" / "2B"
            manifest = root / "trusted-2B.json"
            prepared = PreparedDataset(
                name="fineweb-2b-gpt2",
                root=scaled,
                manifest_path=manifest,
                manifest_sha256="a" * 64,
                train_files=(scaled / "fineweb_train_000001.bin",),
                validation_files=(scaled / "fineweb_val_000000.bin",),
                train_tokens=1_900_000_000,
                validation_tokens=100_000_000,
                validation_prefix_tokens=10_485_760,
            )
            fresh10 = PreparedFresh10(
                name="fresh10-v1",
                root=base,
                manifest_path=root / "fresh10.json",
                manifest_sha256="b" * 64,
                domains=(),
            )
            args = cli.build_parser().parse_args(
                [
                    "prepare",
                    "--non-interactive",
                    "--no-save",
                    "--check-only",
                    "--path",
                    str(base),
                    "--profile",
                    "official",
                    "--dataset",
                    "2B",
                ]
            )
            with (
                patch("rig.cli.repo_root", return_value=root),
                patch(
                    "rig.cli.resolve_preparation_manifest",
                    return_value=manifest,
                ),
                patch("rig.cli.environment_checks", return_value=[]) as checks,
                patch("rig.cli.run_doctor", return_value=[]),
                patch("rig.cli.doctor_ok", return_value=True),
                patch("rig.cli.verify_dataset", return_value=prepared) as verify,
                patch("rig.cli.verify_fresh10", return_value=fresh10) as fresh,
            ):
                self.assertEqual(cli.command_prepare(args), 0)
            self.assertFalse(checks.call_args.kwargs["check_data"])
            verify.assert_called_once_with(manifest, scaled, train_shards=19)
            fresh.assert_called_once_with(base)

    def test_remote_prepare_forwards_named_corpus_to_every_peer(self) -> None:
        config = LocalConfig(
            dataset="4B",
            train_shards=39,
            tpu_vm_count=2,
            tpu_vm_hosts="slice-w-[0-1]",
        ).validate()
        inventory = cli.ClusterInventory(
            host_expression="slice-w-[0-1]",
            hosts=("slice-w-0", "slice-w-1"),
            remote_hosts=("slice-w-1",),
            local_host="slice-w-0",
            artifact_host="slice-w-0",
            reported_hostnames={
                "slice-w-0": "slice-w-0",
                "slice-w-1": "slice-w-1",
            },
        )
        args = cli.build_parser().parse_args(["prepare", "--non-interactive"])
        with patch("rig.cli.run_pdsh") as run:
            cli._run_cluster_prepare(
                config,
                args,
                inventory,
                preparation_type="official",
                root=Path("/repo"),
            )
        remote = run.call_args.args[1]
        self.assertIn("--dataset 4B", remote)
        self.assertIn("--train-shards 39", remote)
        self.assertIn("--profile official", remote)
        self.assertEqual(
            run.call_args.kwargs["timeout"],
            cli._remote_prepare_timeout(config, args, "official"),
        )

    def test_remote_prepare_timeout_scales_with_routed_corpus_bytes(self) -> None:
        args = cli.build_parser().parse_args(["prepare", "--non-interactive"])
        classic = cli._remote_prepare_timeout(LocalConfig(), args, "official")
        two_b = cli._remote_prepare_timeout(
            LocalConfig(dataset="2B"), args, "official"
        )
        eight_b = cli._remote_prepare_timeout(
            LocalConfig(dataset="8B"), args, "official"
        )
        hero = cli._remote_prepare_timeout(
            LocalConfig(dataset="hero"), args, "official"
        )
        self.assertLess(classic, two_b)
        self.assertLess(two_b, eight_b)
        self.assertLess(eight_b, hero)
        self.assertGreaterEqual(hero, 6 * 3600)

    def test_invalid_shard_prefix_is_not_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = cli.build_parser().parse_args(
                [
                    "prepare",
                    "--non-interactive",
                    "--no-doctor",
                    "--no-download",
                    "--path",
                    str(root / "cache"),
                    "--profile",
                    "official",
                    "--dataset",
                    "2B",
                    "--train-shards",
                    "20",
                ]
            )
            with patch("rig.cli.repo_root", return_value=root):
                with self.assertRaisesRegex(DataError, "publishes 19 train shards"):
                    cli.command_prepare(args)
            self.assertFalse((root / ".rig.toml").exists())

    def test_named_corpus_allows_an_honest_partial_prefix(self) -> None:
        config = LocalConfig(dataset="2B", train_shards=1)
        route = cli._route_for_config(config, "official")
        self.assertEqual(route.train_shards, 1)
        self.assertEqual(route.train_capacity, 100_000_000)

    def test_wizard_infers_cloud_tpu_host_expression(self) -> None:
        # data path, preparation profile, dataset, shards, artifacts, host count
        answers = ["", "", "", "", "", "4"] + [""] * 10
        with (
            patch("builtins.input", side_effect=answers),
            patch("rig.cli.infer_host_expression", return_value="slice-w-[0-3]"),
        ):
            result, *_ = cli._prepare_wizard(
                LocalConfig(),
                preparation_type="official",
                run_diagnostics=True,
                require_tpu=True,
                download=True,
                save=True,
            )
        self.assertEqual(result.tpu_vm_count, 4)
        self.assertEqual(result.tpu_vm_hosts, "slice-w-[0-3]")

    def test_cluster_prepare_downloads_on_controller_and_every_peer(self) -> None:
        args = cli.build_parser().parse_args(["prepare", "--non-interactive"])
        config = LocalConfig(
            tpu_vm_count=4,
            tpu_vm_hosts="slice-w-[0-3]",
        )
        inventory = cli.ClusterInventory(
            host_expression=config.tpu_vm_hosts,
            hosts=tuple(f"slice-w-{index}" for index in range(4)),
            remote_hosts=tuple(f"slice-w-{index}" for index in range(1, 4)),
            local_host="slice-w-0",
            artifact_host="slice-w-0",
            reported_hostnames={
                f"slice-w-{index}": f"slice-w-{index}" for index in range(4)
            },
        )
        with patch("rig.cli.run_pdsh") as run:
            cli._run_cluster_prepare(
                config,
                args,
                inventory,
                preparation_type="official",
                root=Path("/repo"),
            )

        self.assertEqual(run.call_args.args[0], inventory.hosts)
        remote = run.call_args.args[1]
        self.assertIn("RIG_CLUSTER_WORKER=1", remote)
        self.assertIn("--profile official", remote)
        self.assertIn("--path /repo/shm", remote)
        self.assertIn("sudo -n chown -R", remote)
        self.assertIn("/dev/shm/.speedrun-cache", remote)

    def test_ram_cache_path_detection_does_not_follow_existing_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "shm").symlink_to("/dev/shm")
            self.assertTrue(cli._uses_repo_shm_cache("shm", root))
            self.assertTrue(cli._uses_repo_shm_cache("/dev/shm", root))
            self.assertFalse(cli._uses_repo_shm_cache("data", root))

    def test_profile_launches_every_configured_host_with_controller_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = root / "recipes" / "variant"
            recipe.mkdir(parents=True)
            (recipe / "train.py").write_text("pass\n", encoding="utf-8")
            (recipe / "config.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (recipe / "dev.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (root / ".rig.toml").write_text("[rig]\n", encoding="utf-8")
            config = LocalConfig(
                data_path="shm",
                default_profile="dev",
                tpu_vm_count=4,
                tpu_vm_hosts="slice-w-[0-3]",
            )
            inventory = cli.ClusterInventory(
                host_expression=config.tpu_vm_hosts,
                hosts=tuple(f"slice-w-{index}" for index in range(4)),
                remote_hosts=tuple(f"slice-w-{index}" for index in range(1, 4)),
                local_host="slice-w-0",
                artifact_host="slice-w-0",
                reported_hostnames={
                    f"slice-w-{index}": f"slice-w-{index}" for index in range(4)
                },
            )
            prepared = PreparedDataset(
                name="dev",
                root=Path("/dev/shm"),
                manifest_path=root / "manifest.json",
                manifest_sha256="a" * 64,
                train_files=(Path("/dev/shm/train.bin"),),
                validation_files=(Path("/dev/shm/val.bin"),),
                train_tokens=100,
                validation_tokens=20,
            )
            args = cli._parse_arguments(
                cli.build_parser(),
                [
                    "profile",
                    "variant",
                    "--output-dir",
                    "profiles/test",
                    "--stop-after-step",
                    "20",
                    "--",
                    "--variant-mode",
                    "trace",
                ]
            )
            with (
                patch("rig.cli.repo_root", return_value=root),
                patch("rig.cli.load_config", return_value=config),
                patch(
                    "rig.cli.resolve_recipe_plan",
                    return_value=RecipePlan(
                        payload={"stop_after_step": 20, "schedule_steps": 200},
                        sha256="a" * 64,
                    ),
                ) as resolve,
                patch("rig.cli.verify_dataset", return_value=prepared),
                patch("rig.cli._probe_configured_cluster", return_value=inventory),
                patch("rig.cli.sync_workspace") as sync,
                patch("rig.cli.run_pdsh") as run,
            ):
                self.assertEqual(cli.command_profile(args), 0)

            self.assertEqual(run.call_args.args[0], inventory.hosts)
            remote = run.call_args.args[1]
            self.assertIn("RIG_DISTRIBUTED=1", remote)
            self.assertIn("RIG_CONTROLLER_HOSTNAME=slice-w-0", remote)
            self.assertIn("--profile dev", remote)
            self.assertIn("--variant-mode trace", remote)
            self.assertIn("--xprof-dir", remote)
            self.assertIn("profiles/test/xprof", remote)
            self.assertIn("--variant-mode", resolve.call_args.kwargs["arguments"])
            sync.assert_called_once()

    def test_report_has_no_admission_knobs(self) -> None:
        # Every successful run is plotted, so there is nothing to tune here.
        report = cli.build_parser().parse_args(["report"])
        self.assertFalse(hasattr(report, "admission_loss"))
        self.assertFalse(hasattr(report, "include_dev"))
        with self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["report", "--include-dev"])
        self.assertFalse(hasattr(LocalConfig(), "report_admission_loss"))
        prepare = cli.build_parser().parse_args(["prepare"])
        target_action = next(
            action
            for action in cli.build_parser()
            ._subparsers._group_actions[0]
            .choices["prepare"]
            ._actions
            if action.dest == "target_loss"
        )
        self.assertIn("smoke/development", target_action.help)
        self.assertIsNone(prepare.target_loss)
        with self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["report", "--admission-loss", "3.9"])

    def test_leaderboard_skips_malformed_cohort_records(self) -> None:
        args = cli.build_parser().parse_args(["leaderboard", "--profile", "dev"])
        malformed = {
            "status": "ok",
            "run_kind": "full",
            "profile": "dev",
            "cohort_id": "a" * 64,
            "cohort": {"cohort_id": "a" * 64},
        }
        output = StringIO()
        with (
            patch("rig.cli.load_config", return_value=LocalConfig()),
            patch("rig.cli.load_records", return_value=[malformed]),
            redirect_stdout(output),
        ):
            self.assertEqual(cli.command_leaderboard(args), 0)
        self.assertIn("No cohort-tagged full dev runs", output.getvalue())

    def test_dataset_provenance_uses_stable_names_not_cache_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "data" / "manifests" / "tiny.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}", encoding="utf-8")
            prepared = PreparedDataset(
                name="tiny",
                root=Path("/dev/shm"),
                manifest_path=manifest,
                manifest_sha256="a" * 64,
                train_files=(Path("/dev/shm/train-1.bin"),),
                validation_files=(Path("/dev/shm/val.bin"),),
                train_tokens=100,
                validation_tokens=20,
                validation_prefix_tokens=16,
            )
            provenance = cli._data_provenance(
                prepared, profile="dev", integrity="sha256", repo=root
            )["dataset"]
        self.assertEqual(provenance["manifest"]["path"], "data/manifests/tiny.json")
        self.assertEqual(
            provenance["manifest"]["sha256"],
            "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        )
        self.assertEqual(provenance["manifest"]["canonical_sha256"], "a" * 64)
        self.assertEqual(provenance["train_files"], ["train-1.bin"])
        self.assertEqual(provenance["validation_prefix_tokens"], 16)
        self.assertNotIn("/dev/shm", str(provenance))

    def test_fresh10_provenance_records_stable_domain_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "data" / "manifests" / "fresh10.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}", encoding="utf-8")
            prepared = PreparedDataset(
                name="tiny",
                root=Path("/dev/shm"),
                manifest_path=manifest,
                manifest_sha256="a" * 64,
                train_files=(Path("/dev/shm/train.bin"),),
                validation_files=(Path("/dev/shm/val.bin"),),
                train_tokens=10,
                validation_tokens=10,
                validation_prefix_tokens=8,
            )
            domain = Fresh10Domain(
                name="science",
                path=Path("/dev/shm/fresh10-science.bin"),
                token_count=8196,
                scored_tokens=8192,
                sha256="b" * 64,
                documents=(),
            )
            fresh10 = PreparedFresh10(
                name="fresh10-v1",
                root=Path("/dev/shm"),
                manifest_path=manifest,
                manifest_sha256="c" * 64,
                domains=(domain,),
            )
            provenance = cli._data_provenance(
                prepared,
                profile="official",
                integrity="sha256",
                repo=root,
                fresh10=fresh10,
            )["fresh10"]
        self.assertEqual(provenance["scored_tokens"], 8192)
        self.assertEqual(provenance["domains"]["science"]["sha256"], "b" * 64)
        self.assertNotIn("/dev/shm", str(provenance))

    def test_verify_recovers_recorded_fresh10_contract(self) -> None:
        record = {
            "provenance": {
                "fresh10": {
                    "domains": {
                        "science": {"scored_tokens": 8_192},
                        "legal": {"scored_tokens": 4_096},
                    }
                }
            }
        }
        self.assertEqual(
            cli._recorded_downstream_tokens(record),
            {"science": 8_192, "legal": 4_096},
        )
        self.assertIsNone(cli._recorded_downstream_tokens({"provenance": {}}))
        with self.assertRaisesRegex(cli.HarnessError, "invalid domain row"):
            cli._recorded_downstream_tokens(
                {
                    "provenance": {
                        "fresh10": {"domains": {"science": {"scored_tokens": 0}}}
                    }
                }
            )

    def test_verify_recovers_training_budget_without_retroactive_default(self) -> None:
        self.assertIsNone(cli._recorded_training_tokens({}))
        self.assertIsNone(
            cli._recorded_training_tokens({"constraints": {"training_tokens": None}})
        )
        self.assertEqual(
            cli._recorded_training_tokens(
                {"constraints": {"training_tokens": 624_984_064}}
            ),
            624_984_064,
        )
        with self.assertRaisesRegex(cli.HarnessError, "training-token"):
            cli._recorded_training_tokens({"constraints": {"training_tokens": 0}})

    def test_verify_recovers_validation_budget_and_checkpoint_retention(self) -> None:
        self.assertIsNone(cli._recorded_validation_tokens({}))
        self.assertEqual(
            cli._recorded_validation_tokens(
                {"constraints": {"validation_tokens": 10_485_760}}
            ),
            10_485_760,
        )
        with self.assertRaisesRegex(cli.HarnessError, "validation-token"):
            cli._recorded_validation_tokens({"constraints": {"validation_tokens": 0}})

        self.assertTrue(cli._record_requires_checkpoint(None))
        self.assertTrue(
            cli._record_requires_checkpoint({"checkpoint": {"retained": True}})
        )
        self.assertFalse(
            cli._record_requires_checkpoint({"checkpoint": {"retained": False}})
        )
        self.assertFalse(cli._record_requires_checkpoint({"checkpoint": None}))
        with self.assertRaisesRegex(cli.HarnessError, "retention state"):
            cli._record_requires_checkpoint({"checkpoint": {"retained": "sometimes"}})


if __name__ == "__main__":
    unittest.main()
