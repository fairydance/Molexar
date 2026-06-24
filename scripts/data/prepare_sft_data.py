#!/usr/bin/env python3
"""
Prepare SFT Training Data from ChEMBL Database

This script converts ChEMBL data into JSON format suitable for SFT training
with protein ESM embedding conditioning.

Input Files:
- chembl_36_activities.clean.chembl_id.csv: molecule_chembl_id, target_chembl_id pairs
- chembl_36_activities.molecule.clean.fragment_selfies: molecule_chembl_id -> Fragment-SELFIES string
- chembl_36_esm_embeddings_esmc_600m.clean.pkl: target_chembl_id -> ESM embedding

Output:
- JSON file with structure:
  [
    {
      "target": "Fragment-SELFIES string",
      "conditions": {
        "prot_seq_esm_emb": [0.1, 0.2, ..., 0.9]
      }
    }
  ]

Usage:
    python prepare_sft_data.py \
        --activities /path/to/chembl_36_activities.clean.chembl_id.csv \
        --molecules /path/to/chembl_36_activities.molecule.clean.fragment_selfies \
        --esm_embeddings /path/to/chembl_36_esm_embeddings_esmc_600m.clean.pkl \
        --output /path/to/sft_data.json \
        --max_samples 10000
"""

import sys
import os
import argparse
import json
import pickle
import csv
from typing import Dict, List, Optional, Set
from loguru import logger
from tqdm import tqdm

try:
    import torch
    import numpy as np
except ImportError:
    logger.error("PyTorch or NumPy not found. Please install: pip install torch numpy")
    sys.exit(1)

def load_activities(activities_path: str) -> List[tuple]:
    """
    Load molecule-target pairs from CSV.
    Expected format: molecule_chembl_id, target_chembl_id
    """
    logger.info(f"Loading activities from: {activities_path}")
    pairs = []
    
    if not os.path.exists(activities_path):
        logger.error(f"File not found: {activities_path}")
        return pairs
    
    with open(activities_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading activities"):
            mol_id = row.get('molecule_chembl_id')
            target_id = row.get('target_chembl_id')
            if mol_id and target_id:
                pairs.append((mol_id, target_id))
    
    logger.info(f"Loaded {len(pairs)} molecule-target pairs")
    return pairs


def load_fragment_selfies_map(fragment_selfies_path: str) -> Dict[str, str]:
    """
    Load molecule_chembl_id -> Fragment-SELFIES string mapping.
    Supports two formats:
    1. CSV: molecule_chembl_id,fragment_selfies_string
    2. Space-separated: {fragment_selfies} {chembl_id}
    """
    logger.info(f"Loading Fragment-SELFIES strings from: {fragment_selfies_path}")
    fragment_selfies_map = {}
    
    if not os.path.exists(fragment_selfies_path):
        logger.error(f"File not found: {fragment_selfies_path}")
        return fragment_selfies_map
    
    # Try to detect format
    with open(fragment_selfies_path, 'r') as f:
        first_line = f.readline().strip()
        f.seek(0)
        
        # Check if it's CSV format
        if ',' in first_line:
            logger.info("Detected CSV format")
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc="Loading Fragment-SELFIES"):
                mol_id = row.get('molecule_chembl_id')
                fragment_selfies = row.get('fragment_selfies', '').strip()
                if mol_id and fragment_selfies:
                    fragment_selfies_map[mol_id] = fragment_selfies
        else:
            # Space-separated format: {fragment_selfies} {chembl_id}
            logger.info("Detected space-separated format")
            for line in tqdm(f, desc="Loading Fragment-SELFIES"):
                line = line.strip()
                if not line:
                    continue
                # Split by last space to separate fragment_selfies and chembl_id
                parts = line.rsplit(' ', 1)
                if len(parts) == 2:
                    fragment_selfies = parts[0].strip()
                    mol_id = parts[1].strip()
                    if mol_id and fragment_selfies:
                        fragment_selfies_map[mol_id] = fragment_selfies
    
    logger.info(f"Loaded {len(fragment_selfies_map)} Fragment-SELFIES strings")
    return fragment_selfies_map


def load_esm_embeddings(esm_path: str) -> Dict[str, torch.Tensor]:
    """
    Load target_chembl_id -> ESM embedding mapping.
    Expected format: pickle file with dict structure
    """
    logger.info(f"Loading ESM embeddings from: {esm_path}")
    esm_map = {}
    
    if not os.path.exists(esm_path):
        logger.error(f"File not found: {esm_path}")
        return esm_map
    
    with open(esm_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different pickle formats
    if isinstance(data, dict):
        for target_id, embedding in tqdm(data.items(), desc="Processing ESM"):
            if isinstance(embedding, torch.Tensor):
                esm_map[target_id] = embedding
            elif isinstance(embedding, list):
                esm_map[target_id] = torch.tensor(embedding, dtype=torch.float32)
            elif hasattr(embedding, 'numpy'):  # numpy array or similar
                esm_map[target_id] = torch.tensor(embedding.numpy(), dtype=torch.float32)
            elif isinstance(embedding, np.ndarray):
                esm_map[target_id] = torch.tensor(embedding, dtype=torch.float32)
            else:
                logger.warning(f"Unexpected embedding type for {target_id}: {type(embedding)}")
    else:
        logger.error(f"Expected dict in pickle file, got {type(data)}")
    
    logger.info(f"Loaded {len(esm_map)} ESM embeddings")
    return esm_map


def prepare_sft_data(
    activities: List[tuple],
    fragment_selfies_map: Dict[str, str],
    esm_map: Dict[str, torch.Tensor],
    max_samples: Optional[int] = None,
    filter_invalid: bool = True
) -> List[Dict]:
    """
    Combine all data into SFT format.
    """
    logger.info("Preparing SFT data...")
    
    sft_data = []
    skipped_invalid = 0
    skipped_missing = 0
    
    # Fixed ESM dimension
    ESM_DIM = 1152
    
    # Use set to avoid duplicates
    processed_pairs: Set[str] = set()
    
    iterator = tqdm(activities, desc="Processing pairs")
    for mol_id, target_id in iterator:
        # Create unique key to avoid duplicates
        pair_key = f"{mol_id}_{target_id}"
        if pair_key in processed_pairs:
            continue
        
        # Check if we have all required data
        if mol_id not in fragment_selfies_map:
            skipped_missing += 1
            continue
        
        if target_id not in esm_map:
            skipped_missing += 1
            continue
        
        fragment_selfies = fragment_selfies_map[mol_id]
        esm_emb = esm_map[target_id]
        
        # Validate ESM embedding dimension
        if esm_emb.dim() != 1:
            logger.warning(f"ESM embedding for {target_id} has wrong shape: {esm_emb.shape}")
            skipped_invalid += 1
            continue
        
        if esm_emb.shape[0] != ESM_DIM:
            logger.warning(f"ESM embedding for {target_id} has wrong dimension: {esm_emb.shape[0]} != {ESM_DIM}")
            skipped_invalid += 1
            continue
        
        # Validate Fragment-SELFIES string
        if filter_invalid and len(fragment_selfies) < 5:
            logger.warning(f"Fragment-SELFIES too short for {mol_id}: {fragment_selfies}")
            skipped_invalid += 1
            continue
        
        # Create SFT entry
        entry = {
            "target": fragment_selfies,
            "conditions": {
                "prot_seq_esm_emb": esm_emb.tolist()
            }
        }
        
        sft_data.append(entry)
        processed_pairs.add(pair_key)
        
        # Stop if we reached max_samples
        if max_samples and len(sft_data) >= max_samples:
            logger.info(f"Reached max_samples limit: {max_samples}")
            break
    
    logger.info(f"Created {len(sft_data)} SFT entries")
    logger.info(f"Skipped (missing data): {skipped_missing}")
    logger.info(f"Skipped (invalid): {skipped_invalid}")
    
    return sft_data


def save_sft_data(sft_data: List[Dict], output_path: str):
    """Save SFT data to JSON file."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving {len(sft_data)} entries to: {output_path}")
    with open(output_path, 'w') as f:
        if output_path.endswith(".jsonl"):
            for item in sft_data:
                f.write(json.dumps(item) + "\n")
        else:
            json.dump(sft_data, f, indent=2)
    
    logger.success(f"Saved SFT data successfully!")


def resolve_data_path(data_dir: str, path: str) -> str:
    if not path or os.path.isabs(path):
        return path
    return os.path.join(data_dir, path)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare protein-sequence conditioning data from activity, molecule, and embedding files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python prepare_sft_data.py \\
        --protein_sequence_dir /path/to/protein_sequence_esmc_600m \\
        --output data/sft_data.jsonl \\
        --max_samples 50000
        """
    )

    parser.add_argument(
        "--protein_sequence_dir",
        type=str,
        required=True,
        help="Directory containing protein-sequence conditioning files"
    )
    
    parser.add_argument(
        "--activities",
        type=str,
        default="chembl_36_activities.clean.chembl_id.csv",
        help="Activities CSV path or filename under --protein_sequence_dir"
    )
    
    parser.add_argument(
        "--molecules",
        type=str,
        default="chembl_36_activities.molecule.clean.fragment_selfies",
        help="Molecule file path or filename under --protein_sequence_dir"
    )
    
    parser.add_argument(
        "--esm_embeddings",
        type=str,
        default="chembl_36_esm_embeddings_esmc_600m.clean.pkl",
        help="ESM embedding PKL path or filename under --protein_sequence_dir"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="chembl_36_protein_sequence_sft.jsonl",
        help="Output JSON or JSONL file path"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to include (default: all)"
    )
    
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="Don't filter invalid entries"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()

    if args.protein_sequence_dir:
        args.activities = resolve_data_path(args.protein_sequence_dir, args.activities)
        args.molecules = resolve_data_path(args.protein_sequence_dir, args.molecules)
        args.esm_embeddings = resolve_data_path(args.protein_sequence_dir, args.esm_embeddings)
        args.output = resolve_data_path(args.protein_sequence_dir, args.output)

    missing_paths = [
        name
        for name, value in (
            ("--activities", args.activities),
            ("--molecules", args.molecules),
            ("--esm_embeddings", args.esm_embeddings),
            ("--output", args.output),
        )
        if not value
    ]
    if missing_paths:
        logger.error(
            "Missing required paths: "
            + ", ".join(missing_paths)
            + ". Pass them explicitly or set --protein_sequence_dir."
        )
        sys.exit(1)

    # Configure logging
    logger.configure(
        handlers=[
            {"sink": sys.stderr, "level": "DEBUG" if args.verbose else "INFO"}
        ]
    )
    
    # Load all data
    activities = load_activities(args.activities)
    if not activities:
        logger.error("No activities loaded")
        sys.exit(1)
    
    fragment_selfies_map = load_fragment_selfies_map(args.molecules)
    if not fragment_selfies_map:
        logger.error("No Fragment-SELFIES strings loaded")
        sys.exit(1)
    
    esm_map = load_esm_embeddings(args.esm_embeddings)
    if not esm_map:
        logger.error("No ESM embeddings loaded")
        sys.exit(1)
    
    # Prepare SFT data
    sft_data = prepare_sft_data(
        activities=activities,
        fragment_selfies_map=fragment_selfies_map,
        esm_map=esm_map,
        max_samples=args.max_samples,
        filter_invalid=not args.no_filter
    )
    
    if not sft_data:
        logger.error("No valid SFT data created")
        sys.exit(1)
    
    # Save
    save_sft_data(sft_data, args.output)
    
    # Summary
    logger.info("="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Total SFT entries: {len(sft_data)}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Sample entry:")
    if sft_data:
        print(json.dumps(sft_data[0], indent=2))
    logger.info("="*60)


if __name__ == "__main__":
    main()
