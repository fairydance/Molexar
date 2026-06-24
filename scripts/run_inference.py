#!/usr/bin/env python3
"""Unified Molexar inference script for base and conditional models."""

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

from loguru import logger

from molexar.inference import DEFAULT_MAX_NEW_TOKENS, MolexarInference, load_conditions_from_args


GENERATION_TASK_ALIASES = {
    "de_novo": "de_novo",
    "denovo": "de_novo",
    "motif_extension": "motif_extension",
    "scaffold_decoration": "scaffold_decoration",
    "linker_design": "linker_design",
    "scaffold_morphing": "scaffold_morphing",
    "superstructure": "superstructure",
    "super_structure": "superstructure",
}
FRAGMENT_CONSTRAINED_TASKS = {
    "motif_extension",
    "scaffold_decoration",
    "linker_design",
    "scaffold_morphing",
    "superstructure",
}
SINGLE_FRAGMENT_TASKS = {"motif_extension", "scaffold_decoration"}
TWO_FRAGMENT_TASKS = {"linker_design", "scaffold_morphing"}
IODINE_SUBSTITUTION_MESSAGE = (
    "The current tokenizer vocabulary does not include the iodine token [I]. "
    "Please substitute iodine (I) with bromine (Br) in the start constraint first."
)


def normalize_generation_task(value):
    normalized = value.strip().lower().replace("-", "_")
    task = GENERATION_TASK_ALIASES.get(normalized)
    if task is None:
        choices = ", ".join(sorted(set(GENERATION_TASK_ALIASES.values())))
        raise argparse.ArgumentTypeError(f"unsupported generation task '{value}'; choices: {choices}")
    return task


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Molexar base and conditional inference")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default="models/tokenizer")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=["base", "conditional"], default="base")

    parser.add_argument("--condition_json", default=None, help="JSON string/path with condition key-value pairs")
    parser.add_argument("--condition_file", default=None, help="JSON/PKL/NPY condition file")
    parser.add_argument("--condition_key", default=None, help="Condition key for NPY files")
    parser.add_argument("--reference_smiles", default=None, help="Reference SMILES for property or pharmacophore-FP conditioning")
    parser.add_argument("--property_keys", default=None, help="Comma-separated property keys to keep from SMILES-derived properties")
    parser.add_argument("--num_random_properties", type=int, default=None, help="Random property subset size for SMILES-derived properties")
    parser.add_argument("--pharma_fp_file", default=None, help="NPY/JSON/PKL pharmacophore fingerprint condition")
    parser.add_argument("--esm_file", default=None, help="NPY/JSON/PKL ESM embedding condition")
    parser.add_argument("--protein_sequence", default=None, help="Protein amino acid sequence for ESM conditioning")
    parser.add_argument("--protein_sequence_file", default=None, help="Text file containing protein sequence for ESM conditioning")
    parser.add_argument("--esm_conda_env", default="molexar_aux", help="Conda environment used to compute ESM embeddings")
    parser.add_argument("--esm_device", choices=["cuda", "cpu"], default="cuda", help="Device used by ESM embedding calculation")
    parser.add_argument(
        "--esm_embedding_script",
        default=None,
        help="Optional path to compute_esm_embedding.py. Defaults to scripts/embeddings/compute_esm_embedding.py.",
    )
    parser.add_argument("--pocket_pdb", default=None, help="PDB file for pocket GVP conditioning")
    for key in [
        "mol_hac",
        "mol_hbdc",
        "mol_hbac",
        "mol_rotbc",
        "mol_wt",
        "mol_logp",
        "mol_tpsa",
        "mol_qed",
        "mol_sas",
    ]:
        parser.add_argument(f"--{key}", type=float, default=None)
    parser.add_argument("--pocket_radius", type=float, default=25.0)
    parser.add_argument("--max_atoms", type=int, default=425)

    parser.add_argument("--start_string", default=None, help="Fragment-SELFIES prefix used after <MOL>")
    parser.add_argument(
        "--start_smiles",
        default=None,
        help="SMILES constraint converted into the Fragment-SELFIES prefix used after <MOL>",
    )
    start_smiles_mode = parser.add_mutually_exclusive_group()
    start_smiles_mode.add_argument(
        "--start_smiles_canonical",
        action="store_true",
        default=False,
        help="Encode --start_smiles canonically before generation.",
    )
    start_smiles_mode.add_argument(
        "--start_smiles_randomized",
        action="store_true",
        default=False,
        help="Encode --start_smiles with randomized Fragment-SELFIES traversal. This is the default.",
    )
    parser.add_argument(
        "--generation_task",
        type=normalize_generation_task,
        default=None,
        help=(
            "Generation task: de_novo, motif_extension, scaffold_decoration, "
            "linker_design, scaffold_morphing, or superstructure"
        ),
    )
    parser.add_argument(
        "--attachment_point_depth",
        type=int,
        default=None,
        help="Maximum number of attachment points to open for superstructure SMILES inputs",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help=(
            f"Maximum generated tokens. Defaults to {DEFAULT_MAX_NEW_TOKENS}, or "
            f"{DEFAULT_MAX_NEW_TOKENS} minus the tokenized resolved start string length."
        ),
    )
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--no_do_sample", action="store_false", dest="do_sample")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--batch_delay", type=float, default=0.0)

    parser.add_argument("--convert_to_smiles", action="store_true")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--canonical", action="store_true", default=False)
    output_mode.add_argument("--randomized", action="store_true", default=False)
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Require strict Fragment-SELFIES decoding. Default is non-strict decoding for generated samples.",
    )
    repair_mode = parser.add_mutually_exclusive_group()
    repair_mode.add_argument(
        "--repair_missing_return_attachment",
        action="store_true",
        dest="repair_missing_return_attachment",
        default=None,
        help=(
            "Repair generated Fragment-SELFIES with missing linker return attachments. "
            "Defaults to enabled for linker_design and scaffold_morphing."
        ),
    )
    repair_mode.add_argument(
        "--no_repair_missing_return_attachment",
        action="store_false",
        dest="repair_missing_return_attachment",
        help="Disable missing linker return-attachment repair.",
    )
    parser.add_argument("--output_file", default=None)
    parser.add_argument(
        "--output_format",
        choices=["json", "jsonl", "csv", "tsv", "smi"],
        default=None,
        help="Output format. Defaults to the --output_file extension, or JSON if extension is unknown.",
    )
    parser.add_argument(
        "--fragment_selfies_file",
        default=None,
        help="Optional text file with one generated Fragment-SELFIES string per line.",
    )
    parser.add_argument(
        "--smiles_file",
        default=None,
        help="Optional text file with one converted SMILES string per line. Requires --convert_to_smiles.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def infer_output_format(output_file, output_format):
    if output_format is not None:
        return output_format
    if output_file is None:
        return "json"
    suffix = Path(output_file).suffix.lower().lstrip(".")
    if suffix == "smiles":
        return "smi"
    if suffix in {"json", "jsonl", "csv", "tsv", "smi"}:
        return suffix
    return "json"


def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def resolve_generation_inputs(args):
    if args.start_string and args.start_smiles:
        raise ValueError("Provide either --start_string or --start_smiles, not both")

    if args.start_smiles:
        if args.generation_task is None:
            raise ValueError("--generation_task is required when --start_smiles is provided")
        if args.generation_task == "de_novo":
            raise ValueError("de_novo generation does not accept --start_smiles")
        validate_start_smiles_supported_atoms(args.start_smiles)
        start_smiles_encoding = resolve_start_smiles_encoding(args)
        start_string = build_start_string_from_smiles(
            args.start_smiles,
            args.generation_task,
            canonical=start_smiles_encoding == "canonical",
            randomized=start_smiles_encoding == "randomized",
            attachment_point_depth=args.attachment_point_depth,
        )
        return start_string, args.generation_task

    if args.start_string:
        if args.generation_task == "de_novo":
            raise ValueError("de_novo generation does not accept --start_string")
        validate_start_string_supported_tokens(args.start_string)
        return args.start_string, args.generation_task or "custom_start_string"

    if args.generation_task in FRAGMENT_CONSTRAINED_TASKS:
        raise ValueError(f"{args.generation_task} requires --start_smiles or --start_string")
    return None, "de_novo"


def resolve_start_smiles_encoding(args) -> str:
    return "canonical" if getattr(args, "start_smiles_canonical", False) else "randomized"


def resolve_repair_missing_return_attachment(args, generation_task: str) -> bool:
    if args.repair_missing_return_attachment is not None:
        return args.repair_missing_return_attachment
    return generation_task in TWO_FRAGMENT_TASKS


def build_start_string_from_smiles(
    smiles: str,
    generation_task: str,
    rng: Optional[random.Random] = None,
    canonical: bool = False,
    randomized: Optional[bool] = None,
    attachment_point_depth: Optional[int] = None,
) -> str:
    randomized = not canonical if randomized is None else randomized
    if canonical and randomized:
        raise ValueError("canonical and randomized start-SMILES encoding are mutually exclusive")
    validate_start_smiles_supported_atoms(smiles)
    rng = rng or random.Random()
    if generation_task in SINGLE_FRAGMENT_TASKS:
        return build_single_fragment_start_string(smiles, rng, canonical, randomized)
    if generation_task in TWO_FRAGMENT_TASKS:
        return build_two_fragment_start_string(smiles, rng, canonical, randomized)
    if generation_task == "superstructure":
        variants = enumerate_attachment_point_variants(smiles, depth=attachment_point_depth)
        if not variants:
            raise ValueError("Could not add any attachment points for superstructure generation")
        return build_single_fragment_start_string(rng.choice(variants), rng, canonical, randomized)
    raise ValueError(f"{generation_task} does not use a SMILES start string")


def build_single_fragment_start_string(
    smiles: str,
    rng: random.Random,
    canonical: bool,
    randomized: bool,
) -> str:
    attachment_count = count_attachment_points(smiles)
    if attachment_count < 1:
        raise ValueError("SMILES input must contain at least one attachment point")
    attach_index = 0 if attachment_count == 1 else rng.randrange(attachment_count)
    return (
        f"{encode_fragment_smiles(smiles, canonical, randomized, next_random_seed(rng))}"
        f"[Attach:{attach_index}]"
    )


def build_two_fragment_start_string(
    smiles: str,
    rng: random.Random,
    canonical: bool,
    randomized: bool,
) -> str:
    fragments = [fragment.strip() for fragment in smiles.split(".") if fragment.strip()]
    if len(fragments) != 2:
        raise ValueError(
            "Linker design/scaffold morphing SMILES must contain exactly two "
            "dot-separated fragments"
        )
    for fragment in fragments:
        attachment_count = count_attachment_points(fragment)
        if attachment_count != 1:
            raise ValueError(
                "Each linker/scaffold-morphing fragment must contain exactly one "
                "attachment point"
            )
    rng.shuffle(fragments)
    return "".join(
        encode_fragment_smiles(fragment, canonical, randomized, next_random_seed(rng))
        for fragment in fragments
    ) + "[Attach:0]"


def next_random_seed(rng: random.Random) -> int:
    return rng.randrange(2**32)


def encode_fragment_smiles(
    smiles: str,
    canonical: bool = False,
    randomized: Optional[bool] = None,
    seed: Optional[int] = None,
) -> str:
    randomized = not canonical if randomized is None else randomized
    if canonical and randomized:
        raise ValueError("canonical and randomized start-SMILES encoding are mutually exclusive")
    validate_start_smiles_supported_atoms(smiles)
    try:
        from molexar.data.converter import smiles_fragment_to_fragment_selfies
    except ImportError as exc:
        raise ImportError(
            "Fragment-SELFIES is required to convert --start_smiles to --start_string"
        ) from exc
    return smiles_fragment_to_fragment_selfies(
        smiles,
        canonical=canonical,
        randomized=randomized,
        seed=seed if randomized else None,
    )


def count_attachment_points(smiles: str) -> int:
    _, mol = parse_smiles(smiles)
    return count_dummy_atoms(mol)


def validate_start_string_supported_tokens(start_string: str) -> None:
    if "[I]" in start_string:
        raise ValueError(IODINE_SUBSTITUTION_MESSAGE)


def validate_start_smiles_supported_atoms(smiles: str) -> None:
    _, mol = parse_smiles(smiles)
    if any(atom.GetAtomicNum() == 53 for atom in mol.GetAtoms()):
        raise ValueError(IODINE_SUBSTITUTION_MESSAGE)


def enumerate_attachment_point_variants(smiles: str, depth: Optional[int] = None):
    Chem, mol = parse_smiles(smiles)
    if count_dummy_atoms(mol):
        raise ValueError("Superstructure SMILES must be a complete molecule without attachment points")
    target_depth = depth or 1
    if target_depth < 1:
        raise ValueError("--attachment_point_depth must be at least 1")

    frontier = [mol]
    variants = {}
    for _ in range(target_depth):
        next_frontier = []
        for current_mol in frontier:
            for opened_mol in open_single_attachment_points(Chem, current_mol):
                opened_smiles = Chem.MolToSmiles(opened_mol, canonical=True, isomericSmiles=False)
                variants[opened_smiles] = opened_mol
                next_frontier.append(opened_mol)
        frontier = unique_molecules(Chem, next_frontier)
        if not frontier:
            break
    return sorted(variants)


def parse_smiles(smiles: str):
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("RDKit is required to convert --start_smiles to --start_string") from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem, mol


def count_dummy_atoms(mol) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0)


@lru_cache(maxsize=1)
def attachment_opening_reaction():
    from rdkit.Chem import AllChem

    return AllChem.ReactionFromSmarts("[*;h;!$([*][#0]):1]>>[*:1][*]")


def open_single_attachment_points(Chem, mol):
    products = []
    reaction = attachment_opening_reaction()
    for product_tuple in reaction.RunReactants((mol,)):
        product = product_tuple[0]
        if Chem.SanitizeMol(product, catchErrors=True):
            continue
        Chem.AssignStereochemistry(product, force=True)
        products.append(product)
    return unique_molecules(Chem, products)


def unique_molecules(Chem, mols):
    unique = {}
    for mol in mols:
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        unique[smiles] = mol
    return list(unique.values())


def repair_generated_fragment_selfies(fragment_selfies_strings):
    try:
        from fragment_selfies import FragmentSelfiesCodec
    except ImportError as exc:
        raise ImportError("Fragment-SELFIES is required for linker return-attachment repair") from exc

    codec = FragmentSelfiesCodec()
    repaired_strings = []
    for fragment_selfies in fragment_selfies_strings:
        try:
            repaired_strings.append(
                codec.reserialize(
                    fragment_selfies,
                    fallback_selfies=False,
                    canonical=True,
                    randomized=False,
                    strict=True,
                    repair_missing_return_attachment=True,
                )
            )
        except Exception as exc:
            logger.warning(
                f"Linker return-attachment repair failed for '{fragment_selfies}': {exc}"
            )
            repaired_strings.append(fragment_selfies)
    return repaired_strings


def indexed_row(index, row):
    item = {"index": index, "fragment_selfies": row["fragment_selfies"]}
    if "smiles" in row:
        item["smiles"] = row.get("smiles") or ""
    return item


def write_delimited(rows, output_file, delimiter):
    fieldnames = ["index", "fragment_selfies"]
    if any("smiles" in row for row in rows):
        fieldnames.append("smiles")
    ensure_parent_dir(output_file)
    with open(output_file, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow(indexed_row(index, row))


def write_jsonl(rows, output_file):
    ensure_parent_dir(output_file)
    with open(output_file, "w") as handle:
        for index, row in enumerate(rows, start=1):
            handle.write(json.dumps(indexed_row(index, row)) + "\n")


def write_smi(rows, output_file):
    ensure_parent_dir(output_file)
    with open(output_file, "w") as handle:
        for row in rows:
            smiles = row.get("smiles")
            if smiles:
                handle.write(f"{smiles}\n")


def write_output(payload, rows, output_file, output_format):
    if output_format == "json":
        ensure_parent_dir(output_file)
        with open(output_file, "w") as handle:
            json.dump(payload, handle, indent=2)
    elif output_format == "jsonl":
        write_jsonl(rows, output_file)
    elif output_format == "csv":
        write_delimited(rows, output_file, delimiter=",")
    elif output_format == "tsv":
        write_delimited(rows, output_file, delimiter="\t")
    elif output_format == "smi":
        write_smi(rows, output_file)
    else:
        raise ValueError(f"unsupported output format: {output_format}")


def write_separate_files(rows, fragment_selfies_file, smiles_file):
    if fragment_selfies_file:
        ensure_parent_dir(fragment_selfies_file)
        with open(fragment_selfies_file, "w") as handle:
            for row in rows:
                handle.write(f"{row['fragment_selfies']}\n")
        logger.success(f"Fragment-SELFIES saved to {fragment_selfies_file}")

    if smiles_file:
        ensure_parent_dir(smiles_file)
        with open(smiles_file, "w") as handle:
            for row in rows:
                smiles = row.get("smiles")
                if smiles:
                    handle.write(f"{smiles}\n")
        logger.success(f"SMILES saved to {smiles_file}")


def print_rows(rows, output_format):
    if output_format == "jsonl":
        for index, row in enumerate(rows, start=1):
            print(json.dumps(indexed_row(index, row)))
    elif output_format in {"csv", "tsv"}:
        fieldnames = ["index", "fragment_selfies"]
        if any("smiles" in row for row in rows):
            fieldnames.append("smiles")
        writer = csv.DictWriter(
            sys.stdout,
            fieldnames=fieldnames,
            delimiter="," if output_format == "csv" else "\t",
        )
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow(indexed_row(index, row))
    elif output_format == "smi":
        for row in rows:
            smiles = row.get("smiles")
            if smiles:
                print(smiles)
    else:
        for index, row in enumerate(rows, start=1):
            if "smiles" in row:
                print(f"{index}\t{row['fragment_selfies']}\t{row.get('smiles') or ''}")
            else:
                print(f"{index}\t{row['fragment_selfies']}")


def main():
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")
    output_format = infer_output_format(args.output_file, args.output_format)
    if (output_format == "smi" or args.smiles_file) and not args.convert_to_smiles:
        raise SystemExit("--output_format smi and --smiles_file require --convert_to_smiles")
    try:
        start_string, generation_task = resolve_generation_inputs(args)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    repair_missing_return_attachment = resolve_repair_missing_return_attachment(args, generation_task)
    if args.start_smiles:
        start_smiles_encoding = resolve_start_smiles_encoding(args)
        logger.info(
            f"Converted {args.generation_task} SMILES to start_string "
            f"using {start_smiles_encoding} encoding: {start_string}"
        )
    if repair_missing_return_attachment:
        logger.info("Generated Fragment-SELFIES linker return-attachment repair is enabled")

    engine = MolexarInference(args.model_path, device=args.device, tokenizer_path=args.tokenizer_path)
    conditions = load_conditions_from_args(args, engine.config) if args.mode == "conditional" else {}
    if args.mode == "conditional" and not conditions:
        logger.warning("Conditional mode selected but no conditions were provided")
    if args.mode == "conditional" and len(conditions) > 3:
        logger.warning(
            f"Received {len(conditions)} conditions. Universal multi-condition SFT is trained with "
            "at most 3 conditions per sample, so 4+ condition generation may be out of distribution."
        )

    fragment_selfies = engine.generate(
        conditions=conditions,
        start_string=start_string,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample and not args.greedy,
        repetition_penalty=args.repetition_penalty,
        batch_size=args.batch_size,
        batch_delay=args.batch_delay,
    )
    if repair_missing_return_attachment:
        fragment_selfies = repair_generated_fragment_selfies(fragment_selfies)

    rows = []
    if args.convert_to_smiles:
        for fs, smiles in engine.convert_to_smiles(
            fragment_selfies,
            canonical=args.canonical,
            randomized=args.randomized,
            strict=args.strict,
        ):
            rows.append({"fragment_selfies": fs, "smiles": smiles})
    else:
        rows = [{"fragment_selfies": item} for item in fragment_selfies]

    payload = {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "generation_task": generation_task,
        "start_string": start_string,
        "repair_missing_return_attachment": repair_missing_return_attachment,
        "conditions": list(conditions.keys()),
        "results": rows,
    }
    if args.start_smiles:
        payload["start_smiles"] = args.start_smiles
        payload["start_smiles_encoding"] = start_smiles_encoding
    if args.output_file:
        write_output(payload, rows, args.output_file, output_format)
        logger.success(f"Results saved to {args.output_file} ({output_format})")
    else:
        print_rows(rows, output_format if args.output_format else "legacy")

    write_separate_files(rows, args.fragment_selfies_file, args.smiles_file)


if __name__ == "__main__":
    main()
