#!/usr/bin/env python3
"""Extract Fragment-SELFIES fragments that can seed Molexar generation."""

import argparse
import csv
import json
import os
import sys
from typing import Dict, Iterable, List, Optional

from loguru import logger

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except ImportError:
    pass

try:
    from molexar.data.start_fragments import (
        detect_input_format,
        encode_molecule,
        extract_start_strings,
        start_string_to_smiles,
    )
except ImportError:
    logger.error(
        "Could not import Molexar or Fragment-SELFIES conversion utilities. "
        "Install both packages in the active environment first."
    )
    sys.exit(1)


STRUCTURED_FIELDS = [
    "line_number",
    "molecule_id",
    "fragment_index",
    "input_format",
    "input",
    "fragment_selfies",
    "start_string",
    "fragment_smiles",
]


def parse_input_line(line: str) -> tuple[str, Optional[str]]:
    parts = line.strip().split(maxsplit=1)
    molecule = parts[0] if parts else ""
    molecule_id = parts[1] if len(parts) > 1 else None
    return molecule, molecule_id


def iter_inputs(args: argparse.Namespace) -> Iterable[tuple[int, str, Optional[str]]]:
    if args.input is not None:
        yield 1, args.input.strip(), None
        return

    with open(args.input_file, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            molecule, molecule_id = parse_input_line(line)
            if molecule:
                yield line_number, molecule, molecule_id


def build_records(args: argparse.Namespace) -> List[Dict[str, object]]:
    records = []
    seen = set()
    inputs = iter_inputs(args)
    if args.input_file and tqdm is not None:
        inputs = tqdm(inputs, total=args.max_molecules, unit="mol", desc="Extracting")

    for input_count, (line_number, molecule_input, molecule_id) in enumerate(inputs, start=1):
        if args.max_molecules is not None and input_count > args.max_molecules:
            break

        try:
            input_format = detect_input_format(molecule_input, args.input_format)
            fragment_selfies = encode_molecule(
                molecule_input,
                input_format,
                canonical=args.canonical,
                randomized=args.randomized,
                seed=args.seed,
                implicit_probability=args.implicit_probability,
                max_implicit_cuts=args.max_implicit_cuts,
                fragment_style=args.fragment_style,
            )
            start_strings = extract_start_strings(fragment_selfies)
        except Exception as exc:
            logger.error(f"Failed to extract fragments at line {line_number}: {exc}")
            continue

        for fragment_index, start_string in enumerate(start_strings, start=1):
            if args.unique and start_string in seen:
                continue
            seen.add(start_string)

            fragment_smiles = ""
            if args.validate:
                try:
                    fragment_smiles = start_string_to_smiles(start_string, strict=args.strict)
                except Exception as exc:
                    logger.warning(f"Skipping invalid start string at line {line_number}: {exc}")
                    continue

            records.append(
                {
                    "line_number": line_number,
                    "molecule_id": molecule_id or "",
                    "fragment_index": fragment_index,
                    "input_format": input_format,
                    "input": molecule_input,
                    "fragment_selfies": fragment_selfies,
                    "start_string": start_string,
                    "fragment_smiles": fragment_smiles,
                }
            )
    return records


def write_records(records: List[Dict[str, object]], output_path: Optional[str], output_format: str) -> None:
    handle = sys.stdout
    should_close = False
    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        handle = open(output_path, "w", newline="", encoding="utf-8")
        should_close = True

    try:
        if output_format == "txt":
            for record in records:
                handle.write(f"{record['start_string']}\n")
        elif output_format == "jsonl":
            for record in records:
                handle.write(json.dumps(record) + "\n")
        elif output_format in {"csv", "tsv"}:
            writer = csv.DictWriter(
                handle,
                fieldnames=STRUCTURED_FIELDS,
                delimiter="," if output_format == "csv" else "\t",
            )
            writer.writeheader()
            writer.writerows(records)
        else:
            raise ValueError(f"unsupported output format: {output_format}")
    finally:
        if should_close:
            handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Fragment-SELFIES fragments that can be passed to run_inference.py --start_string."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="One SMILES or Fragment-SELFIES molecule string")
    source.add_argument("--input_file", help="File with one SMILES or Fragment-SELFIES molecule per line")
    parser.add_argument(
        "--input_format",
        choices=["auto", "smiles", "fragment_selfies"],
        default="auto",
        help="Input format. Auto detects Fragment-SELFIES strings that start with [Frag], [Frag@], or [SELFIES].",
    )
    encoding_mode = parser.add_mutually_exclusive_group()
    encoding_mode.add_argument("--canonical", action="store_true", help="Use canonical Fragment-SELFIES encoding for SMILES input")
    encoding_mode.add_argument("--randomized", action="store_true", help="Use randomized Fragment-SELFIES encoding for SMILES input")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for randomized SMILES input encoding")
    parser.add_argument(
        "--implicit-probability",
        type=float,
        default=None,
        help="Probability that randomized auto-style encoding emits an implicit adjacent-root edge",
    )
    parser.add_argument(
        "--max-implicit-cuts",
        type=int,
        default=None,
        help="Maximum implicit BRICS edges to cut per connected component",
    )
    parser.add_argument(
        "--fragment-style",
        choices=["auto", "explicit", "implicit"],
        default=None,
        help="Fragment-SELFIES style to generate for SMILES input",
    )
    parser.add_argument("--max_molecules", type=int, default=None, help="Limit the number of input molecules processed")
    parser.add_argument("--unique", action="store_true", help="Deduplicate start strings across all processed molecules")
    parser.add_argument("--no_validate", action="store_false", dest="validate", default=True, help="Do not decode extracted start strings")
    parser.add_argument("--strict", action="store_true", default=False, help="Use strict validation decoding")
    parser.add_argument("--output", default=None, help="Output path. Defaults to stdout")
    parser.add_argument("--output_format", choices=["txt", "jsonl", "csv", "tsv"], default="txt")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    if args.input_file and not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        raise SystemExit(1)

    records = build_records(args)
    write_records(records, args.output, args.output_format)
    logger.success(f"Extracted {len(records)} start strings")


if __name__ == "__main__":
    main()
