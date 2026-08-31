#!/usr/bin/env python3
"""Build, attest, inspect, and query this recipe's on-disk infinigram index."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


RECIPE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = RECIPE_DIR.parent.parent
DEFAULT_MANIFEST = (
    REPOSITORY_ROOT / "data" / "manifests" / "fineweb-scaled-gpt2" / "8B.json"
)
PATCH_PATH = (
    RECIPE_DIR
    / "patches"
    / "0001-Add-leave-one-out-infinigram-distillation-sampler.patch"
)
UPSTREAM_REPOSITORY = "https://github.com/honglu2875/ngram"
UPSTREAM_REVISION = "2a73b5ffbe852718dbd4e01ee6abafeb1628c5a7"
PROVENANCE_NAME = "provenance.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing {label}: {path}") from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def parse_size(text: str) -> int:
    normalized = text.strip().upper()
    scale = 1
    if normalized and normalized[-1] in "KMGT":
        scale = {
            "K": 10**3,
            "M": 10**6,
            "G": 10**9,
            "T": 10**12,
        }[normalized[-1]]
        normalized = normalized[:-1]
    value = float(normalized)
    result = int(value * scale)
    if not math.isfinite(value) or result <= 0:
        raise argparse.ArgumentTypeError("size must be finite and positive")
    return result


def manifest_contract(manifest_path: Path) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    manifest = read_object(manifest_path, "dataset manifest")
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, Mapping) or (
        tokenizer.get("name") != "gpt2"
        or tokenizer.get("vocab_size") != 50_257
        or tokenizer.get("document_prefix_token") != 50_256
    ):
        raise ValueError("indexing requires the pinned GPT-2 manifest contract")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("dataset manifest has no file inventory")
    train_entries = [
        entry
        for entry in files
        if isinstance(entry, Mapping) and entry.get("split") == "train"
    ]
    if not train_entries:
        raise ValueError("dataset manifest contains no training shards")
    for entry in train_entries:
        if (
            not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("tokens"), int)
            or not isinstance(entry.get("bytes"), int)
            or not isinstance(entry.get("sha256"), str)
        ):
            raise ValueError("dataset manifest has an incomplete training entry")
    return manifest, train_entries


def verify_sources(
    data_root: Path, train_entries: Sequence[Mapping[str, Any]]
) -> list[Path]:
    paths: list[Path] = []
    for index, entry in enumerate(train_entries, 1):
        path = (data_root / str(entry["path"])).resolve()
        if not path.is_file():
            raise ValueError(f"missing training shard: {path}")
        expected_bytes = int(entry["bytes"])
        if path.stat().st_size != expected_bytes:
            raise ValueError(
                f"bad size for {path.name}: expected {expected_bytes:,}, "
                f"found {path.stat().st_size:,}"
            )
        digest = file_sha256(path)
        if digest != entry["sha256"]:
            raise ValueError(
                f"bad SHA-256 for training shard {index}/{len(train_entries)}: "
                f"{path.name}"
            )
        print(f"verified {index:02d}/{len(train_entries):02d} {path.name}")
        paths.append(path)
    return paths


def ngram_runtime() -> tuple[Any, Mapping[str, Any]]:
    try:
        import ngram
        import ngram._core as ngram_core
        import ngram.index as ngram_index
    except ImportError as error:
        raise ValueError(
            "install the patched ngram package described in this recipe's README"
        ) from error
    if not hasattr(ngram.InfiniGram, "infgram_loo_sample_batch"):
        raise ValueError("installed ngram package lacks the leave-one-out sampler")
    metadata = {
        "ngram_version": str(ngram.__version__),
        "ngram_extension_sha256": file_sha256(Path(ngram_core.__file__)),
        "ngram_index_py_sha256": file_sha256(Path(ngram_index.__file__)),
    }
    return ngram, metadata


def finalize_index(
    index_path: Path,
    manifest_path: Path,
    *,
    source_hashes_verified: bool,
) -> Mapping[str, Any]:
    ngram, runtime = ngram_runtime()
    manifest, train_entries = manifest_contract(manifest_path)
    config = ngram.IndexConfig.load(str(index_path))
    expected_tokens = sum(int(entry["tokens"]) for entry in train_entries)
    required = {"ingest", *(f"sa.{i:04d}" for i in range(config.num_shards))}
    if not required.issubset(config.completed):
        raise ValueError(
            "index is incomplete: missing " + ", ".join(sorted(required - set(config.completed)))
        )
    if config.total_tokens != expected_tokens:
        raise ValueError(
            f"index has {config.total_tokens:,} tokens; manifest train split has "
            f"{expected_tokens:,}"
        )
    if (
        config.format != "ngram-sa-v1"
        or config.token_dtype != "u16"
        or config.doc_sep_token != 50_256
        or config.tokenizer != "gpt2"
        or config.vocab_size > 50_257
    ):
        raise ValueError("index configuration differs from the GPT-2 distillation contract")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "index_format": config.format,
        "index_config_sha256": file_sha256(index_path / "config.json"),
        "dataset_name": manifest.get("name"),
        "dataset_manifest": manifest_path.name,
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "source_hashes_verified": bool(source_hashes_verified),
        "train_files": len(train_entries),
        "train_tokens": expected_tokens,
        "documents": int(config.total_docs),
        "index_shards": int(config.num_shards),
        "ngram_upstream_repository": UPSTREAM_REPOSITORY,
        "ngram_upstream_revision": UPSTREAM_REVISION,
        "ngram_patch_sha256": file_sha256(PATCH_PATH),
        **runtime,
    }
    destination = index_path / PROVENANCE_NAME
    temporary = index_path / f".{PROVENANCE_NAME}.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)
    print(f"wrote {destination}")
    return payload


def cmd_build(args: argparse.Namespace) -> int:
    index_path = args.output.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    _, train_entries = manifest_contract(manifest_path)
    paths = verify_sources(args.data_root.expanduser().resolve(), train_entries)
    ngram, _ = ngram_runtime()
    ngram.build_index(
        [str(path) for path in paths],
        str(index_path),
        fmt="llmc",
        token_dtype="u16",
        doc_sep_token=50_256,
        insert_sep=False,
        shard_tokens=args.shard_tokens,
        cpus=args.cpus,
        mem_gb=args.mem,
        tokenizer_name="gpt2",
        resume=True,
        verbose=True,
    )
    finalize_index(
        index_path, manifest_path, source_hashes_verified=True
    )
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.expanduser().resolve()
    _, train_entries = manifest_contract(manifest_path)
    verified = False
    if args.data_root is not None:
        verify_sources(args.data_root.expanduser().resolve(), train_entries)
        verified = True
    finalize_index(
        args.index.expanduser().resolve(),
        manifest_path,
        source_hashes_verified=verified,
    )
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    ngram, _ = ngram_runtime()
    index_path = args.index.expanduser().resolve()
    index = ngram.InfiniGram(str(index_path), threads=args.threads)
    payload = {
        "path": str(index_path),
        "tokens": int(index.tok_cnt),
        "documents": int(index.doc_cnt),
        "shards": int(index.num_shards),
        "vocab_size": int(index.vocab_size),
        "tokenizer": index.config.tokenizer,
        "document_separator": index.config.doc_sep_token,
        "provenance": read_object(index_path / PROVENANCE_NAME, "index provenance"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    ngram, _ = ngram_runtime()
    index = ngram.InfiniGram(
        str(args.index.expanduser().resolve()), threads=args.threads
    )
    context = [int(token) for token in args.tokens]
    if args.continuation is not None:
        result = index.infgram_prob(
            context, int(args.continuation), max_len=args.max_context
        )
        payload = {
            "continuation": int(args.continuation),
            "probability": float(result.prob),
            "suffix_length": int(result.suffix_len),
            "prompt_count": int(result.prompt_count),
            "continuation_count": int(result.cont_count),
        }
    else:
        result = index.infgram_ntd(context, max_len=args.max_context)
        payload = {
            "suffix_length": int(result.suffix_len),
            "prompt_count": int(result.prompt_count),
            "top": [
                {"token": token, "count": count, "probability": probability}
                for token, count, probability in result.top(args.top)
            ],
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="verify data, build, and attest an index")
    build.add_argument("--data-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    build.add_argument("--shard-tokens", type=parse_size, default=4_000_000_000)
    build.add_argument("--cpus", type=int, default=None)
    build.add_argument("--mem", type=float, default=None, help="memory budget in GiB")
    build.set_defaults(func=cmd_build)

    finalize = commands.add_parser(
        "finalize", help="validate a completed index and write provenance"
    )
    finalize.add_argument("index", type=Path)
    finalize.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    finalize.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="also re-hash every source shard before attesting",
    )
    finalize.set_defaults(func=cmd_finalize)

    info = commands.add_parser("info", help="print index and provenance metadata")
    info.add_argument("index", type=Path)
    info.add_argument("--threads", type=int, default=1)
    info.set_defaults(func=cmd_info)

    query = commands.add_parser("query", help="query by GPT-2 token ids")
    query.add_argument("index", type=Path)
    query.add_argument("tokens", nargs="+", type=int, help="context token ids")
    query.add_argument("--continuation", type=int, default=None)
    query.add_argument("--top", type=int, default=10)
    query.add_argument("--max-context", type=int, default=0)
    query.add_argument("--threads", type=int, default=1)
    query.set_defaults(func=cmd_query)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.func(args))
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
