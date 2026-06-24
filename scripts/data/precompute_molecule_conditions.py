#!/usr/bin/env python3
"""Precompute Molexar molecule-property and Gobbi pharmacophore condition files."""

import argparse
import csv
import importlib.util
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger

try:
    from rdkit import Chem, RDConfig
    from rdkit.Chem import DataStructs, Descriptors, QED
    from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
except ImportError as exc:
    print(f"RDKit is required for molecule-condition precomputation: {exc}", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


_SIG_FACTORY = None
_SA_SCORER = None
_FP_DIM = 0
_PACKED_DIM = 0


def load_sa_scorer():
    scorer_path = Path(RDConfig.RDContribDir) / "SA_Score" / "sascorer.py"
    spec = importlib.util.spec_from_file_location("rdkit_sa_scorer", scorer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load RDKit SA scorer from {scorer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_gobbi_factory():
    factory = Gobbi_Pharm2D.factory
    factory.SetBins([(0, 3), (3, 8)])
    factory.minPointCount = 2
    factory.maxPointCount = 3
    factory.Init()
    return factory


def get_gobbi_fingerprint_size() -> int:
    return configure_gobbi_factory().GetSigSize()


def init_worker(fp_dim: int, packed_dim: int) -> None:
    global _SIG_FACTORY, _SA_SCORER, _FP_DIM, _PACKED_DIM
    _SIG_FACTORY = configure_gobbi_factory()
    _SA_SCORER = load_sa_scorer()
    _FP_DIM = fp_dim
    _PACKED_DIM = packed_dim


def parse_smiles_line(line: str) -> Tuple[Optional[str], Optional[str]]:
    parts = line.strip().split(maxsplit=1)
    if not parts:
        return None, None
    molecule_id = parts[1] if len(parts) == 2 else str(parse_smiles_line.line_number)
    return parts[0], molecule_id


parse_smiles_line.line_number = 0


def pack_on_bits(on_bits) -> np.ndarray:
    fp_packed = np.zeros(_PACKED_DIM, dtype=np.uint8)
    for bit in on_bits:
        if bit < _FP_DIM:
            fp_packed[bit // 8] |= 1 << (7 - bit % 8)
    return fp_packed


def empty_result(line_num: int, smiles: str, molecule_id: str, error: str) -> Dict[str, object]:
    return {
        "line_num": line_num,
        "smiles": smiles,
        "molecule_id": molecule_id,
        "properties": None,
        "fp": np.zeros(_PACKED_DIM, dtype=np.uint8),
        "error": error,
    }


def process_line(item: Tuple[int, str]) -> Dict[str, object]:
    line_num, line = item
    parse_smiles_line.line_number = line_num
    smiles, molecule_id = parse_smiles_line(line)
    if not smiles or not molecule_id:
        return empty_result(line_num, smiles or "", molecule_id or "", "empty line")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return empty_result(line_num, smiles, molecule_id, "invalid smiles")

        properties = {
            "HAC": int(mol.GetNumHeavyAtoms()),
            "HBDC": int(Descriptors.NumHDonors(mol)),
            "HBAC": int(Descriptors.NumHAcceptors(mol)),
            "WT": float(Descriptors.MolWt(mol)),
            "LOGP": float(Descriptors.MolLogP(mol)),
            "TPSA": float(Descriptors.TPSA(mol)),
            "QED": float(QED.qed(mol)),
            "SAS": float(_SA_SCORER.calculateScore(mol)),
        }
        fp = Generate.Gen2DFingerprint(mol, _SIG_FACTORY, dMat=None)
        return {
            "line_num": line_num,
            "smiles": smiles,
            "molecule_id": molecule_id,
            "properties": properties,
            "fp": pack_on_bits(fp.GetOnBits()),
            "error": "",
        }
    except Exception as exc:
        return empty_result(line_num, smiles, molecule_id, str(exc))


def count_nonempty_lines(path: Path) -> int:
    with open(path, "rb") as handle:
        return sum(1 for line in handle if line.strip())


def iter_nonempty_lines(path: Path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line_num, line in enumerate(handle, 1):
            if line.strip():
                yield line_num, line


def iter_progress(iterator, total: int):
    if tqdm is None:
        return iterator
    return tqdm(iterator, total=total, desc="Precomputing", unit="mol")


def write_outputs(args, fp_dim: int, packed_dim: int) -> Dict[str, int]:
    input_path = Path(args.input)
    properties_path = Path(args.properties_output)
    pharma_fp_path = Path(args.pharma_fp_output)
    properties_path.parent.mkdir(parents=True, exist_ok=True)
    pharma_fp_path.parent.mkdir(parents=True, exist_ok=True)

    total = count_nonempty_lines(input_path)
    output_array = np.lib.format.open_memmap(
        pharma_fp_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total, packed_dim),
    )
    fieldnames = [
        "mol_id", "SMILES", "HAC", "HBDC", "HBAC", "WT",
        "LOGP", "TPSA", "QED", "SAS", "valid", "error",
    ]
    stats = {"total": total, "success": 0, "failed": 0}

    with open(properties_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        pool = Pool(args.workers, initializer=init_worker, initargs=(fp_dim, packed_dim))
        try:
            results = pool.imap(process_line, iter_nonempty_lines(input_path), chunksize=args.chunksize)
            for row_index, result in enumerate(iter_progress(results, total)):
                props = result["properties"] or {}
                valid = int(result["properties"] is not None)
                writer.writerow({
                    "mol_id": result["molecule_id"],
                    "SMILES": result["smiles"],
                    "HAC": props.get("HAC", ""),
                    "HBDC": props.get("HBDC", ""),
                    "HBAC": props.get("HBAC", ""),
                    "WT": props.get("WT", ""),
                    "LOGP": props.get("LOGP", ""),
                    "TPSA": props.get("TPSA", ""),
                    "QED": props.get("QED", ""),
                    "SAS": props.get("SAS", ""),
                    "valid": valid,
                    "error": result["error"],
                })
                output_array[row_index] = result["fp"]
                if valid:
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
        finally:
            pool.close()
            pool.join()

    output_array.flush()
    metadata = {
        "input": str(input_path),
        "properties_output": str(properties_path),
        "pharma_fp_output": str(pharma_fp_path),
        "fingerprint_dim": fp_dim,
        "packed_dim": packed_dim,
        **stats,
    }
    metadata_path = (
        Path(args.metadata_output)
        if args.metadata_output
        else properties_path.with_suffix(".metadata.json")
    )
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute Molexar molecule condition files from SMILES")
    parser.add_argument("--input", "-i", required=True, help="Input SMILES file: '<smiles> <molecule_id>'")
    parser.add_argument("--properties_output", required=True, help="Output property CSV")
    parser.add_argument("--pharma_fp_output", required=True, help="Output packed Gobbi pharmacophore FP .npy")
    parser.add_argument("--metadata_output", default=None, help="Optional metadata JSON output")
    parser.add_argument("--workers", "-w", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--chunksize", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be at least 1")
    fp_dim = get_gobbi_fingerprint_size()
    packed_dim = (fp_dim + 7) // 8
    logger.info(f"Input: {args.input}")
    logger.info(f"Properties output: {args.properties_output}")
    logger.info(f"Pharma FP output: {args.pharma_fp_output}")
    logger.info(f"Gobbi fingerprint dim={fp_dim}, packed_dim={packed_dim}")
    stats = write_outputs(args, fp_dim, packed_dim)
    logger.success(f"Precomputed molecule conditions: {stats}")


if __name__ == "__main__":
    main()
