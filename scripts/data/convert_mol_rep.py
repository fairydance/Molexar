#!/usr/bin/env python3
import argparse
import multiprocessing
import os
import sys

from loguru import logger

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from molexar.data.converter import (
        fragment_selfies_fragment_to_smiles,
        fragment_selfies_to_smiles,
        smiles_fragment_to_fragment_selfies,
        smiles_to_fragment_selfies,
    )
except ImportError:
    logger.error(
        "Could not import molexar.data.converter. Please ensure Molexar and "
        "Fragment-SELFIES are installed in the active environment."
    )
    sys.exit(1)


def convert_worker(task_data):
    """Convert one input line between SMILES and Fragment-SELFIES."""
    line_num, raw_str, mode, opts, keep_id = task_data
    content = raw_str.strip()

    if not content:
        return (line_num, content, None, "Empty line", None)

    parts = content.split(maxsplit=1)
    molecule_input = parts[0]
    original_id = parts[1] if len(parts) > 1 else None

    try:
        if mode == "smiles2fragment_selfies":
            if opts.get("fragment", False):
                result = smiles_fragment_to_fragment_selfies(
                    molecule_input,
                    canonical=opts.get("canonical", False),
                    randomized=opts.get("randomized", False),
                    seed=opts.get("seed"),
                )
            else:
                result = smiles_to_fragment_selfies(
                    molecule_input,
                    canonical=opts.get("canonical", False),
                    randomized=opts.get("randomized", False),
                    seed=opts.get("seed"),
                    fallback_selfies=opts.get("fallback_selfies", False),
                    implicit_probability=opts.get("implicit_probability"),
                    max_implicit_cuts=opts.get("max_implicit_cuts"),
                    fragment_style=opts.get("fragment_style"),
                )
        elif mode == "fragment_selfies2smiles":
            if opts.get("fragment", False):
                result = fragment_selfies_fragment_to_smiles(
                    molecule_input,
                    canonical=opts.get("canonical", False),
                    randomized=opts.get("randomized", False),
                )
            else:
                result = fragment_selfies_to_smiles(
                    molecule_input,
                    canonical=opts.get("canonical", False),
                    strict=opts.get("strict", False),
                    randomized=opts.get("randomized", False),
                    ignore_errors=False,
                )
            if result is None:
                raise ValueError("Decoder returned None")
        else:
            return (line_num, content, None, f"Unknown mode: {mode}", None)

        return (line_num, content, result, None, original_id)
    except Exception as exc:
        return (line_num, content, None, str(exc), original_id)


def count_lines(file_path):
    with open(file_path, "rb") as handle:
        return sum(1 for _ in handle)


def iter_tasks(file_path, mode, opts, keep_id):
    with open(file_path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            yield (idx, line, mode, opts, keep_id)


def main():
    parser = argparse.ArgumentParser(description="Batch conversion for SMILES and Fragment-SELFIES.")

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to input file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output/converted_molecules.txt",
        help="Path to output file (default: %(default)s).",
    )
    parser.add_argument("--log", help="Path to log file. If not specified, logs to stderr only.")
    parser.add_argument(
        "-m",
        "--mode",
        default="smiles2fragment_selfies",
        choices=["smiles2fragment_selfies", "fragment_selfies2smiles"],
        help="Conversion mode (default: %(default)s).",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel workers (default: all cores).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100,
        help="Number of lines sent to each worker per batch (default: 100).",
    )
    conversion_mode = parser.add_mutually_exclusive_group()
    conversion_mode.add_argument("--canonical", action="store_true", help="Use canonical encoding or SMILES output.")
    conversion_mode.add_argument("--randomized", action="store_true", help="Use randomized encoding or SMILES output.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--implicit-probability",
        type=float,
        default=None,
        help=(
            "Probability that randomized auto-style encoding emits an implicit adjacent-root edge. "
            "Defaults to Fragment-SELFIES codec behavior."
        ),
    )
    parser.add_argument(
        "--max-implicit-cuts",
        type=int,
        default=None,
        help="Maximum implicit BRICS edges to cut per connected component.",
    )
    parser.add_argument(
        "--fragment-style",
        choices=["auto", "explicit", "implicit"],
        default=None,
        help="Fragment-SELFIES style to generate (default: codec behavior).",
    )
    parser.add_argument(
        "--fallback-selfies",
        action="store_true",
        help="Allow official SELFIES fallback blocks if compact fragment encoding fails.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Require strict Fragment-SELFIES decoding. Default is non-strict decoding.",
    )
    parser.add_argument(
        "--fragment",
        action="store_true",
        help="Preserve dummy atoms as fragment attachment points.",
    )
    parser.add_argument(
        "--no-id",
        action="store_false",
        dest="keep_id",
        default=True,
        help="Do not preserve IDs from input (format: {molecule} {id}).",
    )
    parser.add_argument("--id-sep", default=" ", help="Separator between output and ID (default: space)")

    args = parser.parse_args()

    handlers = [
        {
            "sink": sys.stderr,
            "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        }
    ]

    if args.log:
        handlers.append(
            {
                "sink": args.log,
                "rotation": "10 MB",
                "level": "INFO",
                "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            }
        )

    logger.configure(handlers=handlers)

    if tqdm is None:
        logger.warning("'tqdm' is not installed; progress bars are disabled.")

    opts = {
        "canonical": args.canonical,
        "seed": args.seed,
        "randomized": args.randomized,
        "fallback_selfies": args.fallback_selfies,
        "implicit_probability": args.implicit_probability,
        "max_implicit_cuts": args.max_implicit_cuts,
        "fragment_style": args.fragment_style,
        "strict": args.strict,
        "fragment": args.fragment,
    }

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    if args.workers < 1:
        logger.error("--workers must be at least 1")
        sys.exit(1)
    if args.chunksize < 1:
        logger.error("--chunksize must be at least 1")
        sys.exit(1)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info("Counting input lines...")
    total_lines = count_lines(args.input)
    logger.info(f"Found {total_lines} lines in {args.input}. Processing with {args.workers} workers...")

    success_count = 0
    fail_count = 0
    empty_count = 0

    logger.info("Writing output file...")
    try:
        with open(args.output, "w", encoding="utf-8") as output_handle:
            tasks = iter_tasks(args.input, args.mode, opts, args.keep_id)
            with multiprocessing.Pool(processes=args.workers) as pool:
                process_iterator = pool.imap(convert_worker, tasks, chunksize=args.chunksize)
                if tqdm:
                    process_iterator = tqdm(process_iterator, total=total_lines, unit="lines", desc="Converting")
                for line_num, original_input, converted_output, error_msg, original_id in process_iterator:
                    if error_msg == "Empty line":
                        empty_count += 1
                        continue

                    if converted_output is not None:
                        if args.keep_id and original_id:
                            output_line = f"{converted_output}{args.id_sep}{original_id}"
                        else:
                            output_line = converted_output
                        output_handle.write(output_line + "\n")
                        success_count += 1
                    else:
                        fail_count += 1
                        clean_input = original_input.strip()
                        logger.error(f"Conversion failed at line {line_num}: {error_msg} | Input: '{clean_input}'")
    except IOError as exc:
        logger.critical(f"Failed to write output file: {exc}")
        sys.exit(1)

    logger.info("Processing complete.")
    logger.info(f"Total lines: {total_lines}")
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed: {fail_count}")
    if empty_count > 0:
        logger.info(f"Empty/skipped: {empty_count}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
