#!/usr/bin/env python3
"""Precompute SAIR/PLINDER target-context metadata caches for universal SFT."""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from loguru import logger

try:
    from molexar.modeling import MolexarConfig
    from molexar.training import build_target_context_records
except ImportError as exc:
    print(
        "Molexar must be installed before precomputing target-context caches. "
        "Install with: python -m pip install -e /path/to/Molexar --no-deps",
        file=sys.stderr,
    )
    print(f"Import error: {exc}", file=sys.stderr)
    sys.exit(1)


DEFAULT_DATA_ROOT = os.environ.get("DATA_ROOT", "/path/to/datasets")
DEFAULT_HF_DATA_ROOT = os.environ.get("HF_DATA_ROOT", "/path/to/huggingface/datasets")
DEFAULT_SAIR_ROOT = f"{DEFAULT_HF_DATA_ROOT}/SandboxAQ/SAIR"
DEFAULT_PLINDER_ROOT = f"{DEFAULT_DATA_ROOT}/PLINDER/2024-06/v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute target-context caches used by universal multi-condition SFT."
    )
    parser.add_argument(
        "--base_model",
        default=None,
        help="Base/final model directory used to load MolexarConfig condition dimensions.",
    )
    parser.add_argument(
        "--config_path",
        default=None,
        help="Optional config.json or model directory. Overrides --base_model for config loading.",
    )
    parser.add_argument(
        "--target_context_sources",
        default="sair,plinder",
        help="Comma-separated sources to cache: sair, plinder, or sair,plinder.",
    )
    parser.add_argument(
        "--target_context_cache_dir",
        default="tmp/cache/target_context",
        help="Output cache directory. Must match training's --target_context_cache_dir.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional max target pair records, matching training's --max_samples semantics.",
    )
    parser.add_argument(
        "--verify_target_pocket_paths",
        action="store_true",
        default=True,
        help="Verify pocket files while building cache. Must match training to reuse the cache.",
    )
    parser.add_argument(
        "--no_verify_target_pocket_paths",
        action="store_false",
        dest="verify_target_pocket_paths",
        help="Disable pocket path verification while building cache.",
    )
    parser.add_argument("--sair_index_dir", default=f"{DEFAULT_SAIR_ROOT}/index")
    parser.add_argument("--sair_structures_dir", default=f"{DEFAULT_SAIR_ROOT}/structures_processed")
    parser.add_argument("--sair_pairs_file", default=None)
    parser.add_argument("--sair_ligand_fragment_selfies_file", default=None)
    parser.add_argument("--plinder_root", default=DEFAULT_PLINDER_ROOT)
    parser.add_argument("--plinder_index_dir", default=f"{DEFAULT_PLINDER_ROOT}/index")
    parser.add_argument("--plinder_pairs_file", default=None)
    parser.add_argument("--plinder_ligand_fragment_selfies_file", default=None)
    parser.add_argument("--target_ligand_fragment_selfies_folds", type=int, default=10)
    parser.add_argument(
        "--summary_output",
        default=None,
        help=(
            "Optional summary JSON path. Defaults to "
            "<cache_dir>/target_context_cache_summary.json."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> MolexarConfig:
    config_path = args.config_path or args.base_model
    if config_path:
        logger.info(f"Loading config from: {config_path}")
        return MolexarConfig.from_pretrained(config_path)
    logger.warning("No --base_model or --config_path provided; using default MolexarConfig")
    return MolexarConfig()


def cache_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        target_context_ratio=1,
        target_context_sources=args.target_context_sources,
        target_context_cache_dir=args.target_context_cache_dir,
        max_samples=args.max_samples,
        verify_target_pocket_paths=args.verify_target_pocket_paths,
        sair_index_dir=args.sair_index_dir,
        sair_structures_dir=args.sair_structures_dir,
        sair_pairs_file=args.sair_pairs_file,
        sair_ligand_fragment_selfies_file=args.sair_ligand_fragment_selfies_file,
        plinder_root=args.plinder_root,
        plinder_index_dir=args.plinder_index_dir,
        plinder_pairs_file=args.plinder_pairs_file,
        plinder_ligand_fragment_selfies_file=args.plinder_ligand_fragment_selfies_file,
        target_ligand_fragment_selfies_folds=args.target_ligand_fragment_selfies_folds,
    )


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total_records": len(records),
        "total_pairs": len(records),
        "sources": {},
    }
    for record in records:
        source = record.get("source", "unknown")
        source_summary = summary["sources"].setdefault(
            source,
            {
                "records": 0,
                "records_with_esm": 0,
                "pairs": 0,
                "pairs_with_pocket_path": 0,
            },
        )
        pair = record.get("pair", {})
        source_summary["records"] += 1
        source_summary["records_with_esm"] += int(record.get("esm_embedding") is not None)
        source_summary["pairs"] += 1
        source_summary["pairs_with_pocket_path"] += int(bool(pair.get("pocket_path")))
    return summary


def list_cache_files(cache_dir: str) -> List[str]:
    path = Path(cache_dir)
    if not path.exists():
        return []
    return sorted(str(item) for item in path.glob("*target_context_records*.pkl"))


def write_summary(args: argparse.Namespace, payload: Dict[str, Any]) -> Path:
    output_path = (
        Path(args.summary_output)
        if args.summary_output
        else Path(args.target_context_cache_dir) / "target_context_cache_summary.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples must be positive when provided")
    if args.target_ligand_fragment_selfies_folds < 1:
        raise ValueError("--target_ligand_fragment_selfies_folds must be positive")

    Path(args.target_context_cache_dir).mkdir(parents=True, exist_ok=True)
    config = load_config(args)
    records, lookups = build_target_context_records(cache_args(args), config)
    summary = summarize_records(records)
    payload = {
        "target_context_sources": args.target_context_sources,
        "target_context_cache_dir": args.target_context_cache_dir,
        "max_samples": args.max_samples,
        "verify_target_pocket_paths": args.verify_target_pocket_paths,
        "sair_index_dir": args.sair_index_dir,
        "sair_structures_dir": args.sair_structures_dir,
        "sair_pairs_file": args.sair_pairs_file,
        "sair_ligand_fragment_selfies_file": args.sair_ligand_fragment_selfies_file,
        "plinder_root": args.plinder_root,
        "plinder_index_dir": args.plinder_index_dir,
        "plinder_pairs_file": args.plinder_pairs_file,
        "plinder_ligand_fragment_selfies_file": args.plinder_ligand_fragment_selfies_file,
        "target_ligand_fragment_selfies_folds": args.target_ligand_fragment_selfies_folds,
        "molecule_condition_lookup_sources": sorted(lookups.keys()),
        "cache_files": list_cache_files(args.target_context_cache_dir),
        "summary": summary,
    }
    summary_path = write_summary(args, payload)

    logger.success(
        "Precomputed target-context cache: "
        f"records={summary['total_records']}, pairs={summary['total_pairs']}, "
        f"cache_dir={args.target_context_cache_dir}"
    )
    logger.info(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Target-context cache precomputation failed")
        sys.exit(1)
