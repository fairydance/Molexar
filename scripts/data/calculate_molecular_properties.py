#!/usr/bin/env python3
"""
Calculate Molecular Properties from SMILES

Reads a SMILES file and calculates molecular properties for each molecule.
Input format: Either {smiles} or {smiles} {mol_id}
Output format: CSV with calculated properties

Usage:
    python calculate_molecular_properties.py --input input.smi --output output.csv
    
    # Exclude mol_id column
    python calculate_molecular_properties.py --input input.smi --output output.csv --no-mol-id
    
    # Exclude smiles column
    python calculate_molecular_properties.py --input input.smi --output output.csv --no-smiles
    
    # Exclude both
    python calculate_molecular_properties.py --input input.smi --output output.csv --no-mol-id --no-smiles
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List
from loguru import logger


def _load_sa_scorer():
    from rdkit import RDConfig

    scorer_path = Path(RDConfig.RDContribDir) / "SA_Score" / "sascorer.py"
    spec = importlib.util.spec_from_file_location("rdkit_sa_scorer", scorer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load RDKit SA scorer from {scorer_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import QED

    sascorer = _load_sa_scorer()
except ImportError as e:
    logger.error(f"RDKit not available: {e}")
    logger.error("Please install RDKit from conda-forge: conda install -c conda-forge rdkit")
    sys.exit(1)


def calc_sa_score(mol) -> float:
    """
    Calculate the RDKit Contrib synthetic accessibility score.
    """
    return float(sascorer.calculateScore(mol))


def parse_smiles_line(line: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse a SMILES line which can be either:
    - {smiles}
    - {smiles} {mol_id}
    
    Returns:
        Tuple of (smiles, mol_id)
    """
    line = line.strip()
    if not line:
        return None, None
    
    parts = line.split(maxsplit=1)
    
    if len(parts) == 1:
        # Only SMILES, no mol_id
        return parts[0], None
    else:
        # Both SMILES and mol_id
        return parts[0], parts[1]


def calculate_properties(smiles: str) -> Optional[dict]:
    """
    Calculate molecular properties for a SMILES string.
    
    Returns:
        Dictionary of properties or None if calculation fails
    """
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Calculate properties
        hac = mol.GetNumHeavyAtoms()
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        qed = QED.qed(mol)
        sa_score = calc_sa_score(mol)
        
        return {
            'heavy_atom_count': hac,
            'molecular_weight': mw,
            'logp': logp,
            'hbd': hbd,
            'hba': hba,
            'rotatable_bonds': rot_bonds,
            'qed': qed,
            'sa_score': sa_score
        }
        
    except Exception as e:
        logger.debug(f"Error calculating properties for {smiles}: {e}")
        return None


def process_molecules(
    input_path: Path,
    output_path: Path,
    include_mol_id: bool = True,
    include_smiles: bool = True,
    verbose: bool = False
) -> dict:
    """
    Process molecules from input file and save properties to output CSV.
    
    Returns:
        Statistics dictionary
    """
    logger.info(f"Processing molecules from: {input_path}")
    logger.info(f"Output file: {output_path}")
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"File not found: {input_path}")
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine columns
    columns = []
    if include_mol_id:
        columns.append('mol_id')
    if include_smiles:
        columns.append('smiles')
    columns.extend([
        'heavy_atom_count', 'molecular_weight', 'logp', 
        'hbd', 'hba', 'rotatable_bonds', 'qed', 'sa_score'
    ])
    
    stats = {
        'total': 0,
        'success': 0,
        'failed': 0,
        'no_mol_id': 0
    }
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f_in:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                # Write header
                f_out.write(','.join(columns) + '\n')
                
                # Process each line
                for line_num, line in enumerate(f_in, 1):
                    if not line.strip():
                        continue
                    
                    stats['total'] += 1
                    
                    # Parse line
                    smiles, mol_id = parse_smiles_line(line)
                    
                    if smiles is None:
                        stats['failed'] += 1
                        logger.warning(f"Line {line_num}: Invalid format, skipping")
                        continue
                    
                    # Calculate properties
                    props = calculate_properties(smiles)
                    
                    if props is None:
                        stats['failed'] += 1
                        logger.warning(f"Line {line_num}: Failed to calculate properties for {smiles}")
                        continue
                    
                    # Build output row
                    row = []
                    if include_mol_id:
                        if mol_id is None:
                            stats['no_mol_id'] += 1
                            # Use line number as placeholder if no mol_id
                            mol_id = f"mol_{line_num}"
                        row.append(mol_id)
                    
                    if include_smiles:
                        row.append(smiles)
                    
                    # Add calculated properties
                    row.extend([
                        str(props['heavy_atom_count']),
                        f"{props['molecular_weight']:.2f}",
                        f"{props['logp']:.2f}",
                        str(props['hbd']),
                        str(props['hba']),
                        str(props['rotatable_bonds']),
                        f"{props['qed']:.3f}",
                        f"{props['sa_score']:.3f}"
                    ])
                    
                    f_out.write(','.join(row) + '\n')
                    stats['success'] += 1
                    
                    # Progress logging
                    if verbose and stats['total'] % 1000 == 0:
                        logger.info(f"Processed: {stats['total']:,} | Success: {stats['success']:,} | Failed: {stats['failed']:,}")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("Property Calculation Complete!")
        logger.info("="*60)
        logger.info(f"Total processed: {stats['total']:,}")
        logger.info(f"Success: {stats['success']:,}")
        logger.info(f"Failed: {stats['failed']:,}")
        if stats['total'] > 0:
            logger.info(f"Success rate: {stats['success']/stats['total']*100:.2f}%")
        if stats['no_mol_id'] > 0:
            logger.info(f"Lines without mol_id: {stats['no_mol_id']:,} (assigned generated IDs)")
        logger.info("="*60)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Calculate molecular properties from SMILES file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python calculate_molecular_properties.py --input input.smi --output properties.csv
  
  # Exclude mol_id column
  python calculate_molecular_properties.py --input input.smi --output properties.csv --no-mol-id
  
  # Exclude smiles column
  python calculate_molecular_properties.py --input input.smi --output properties.csv --no-smiles
  
  # Exclude both mol_id and smiles
  python calculate_molecular_properties.py --input input.smi --output properties.csv --no-mol-id --no-smiles
  
  # Verbose mode
  python calculate_molecular_properties.py --input input.smi --output properties.csv --verbose
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input SMILES file (format: {smiles} or {smiles} {mol_id})"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="output/molecular_properties.csv",
        help="Output CSV file (default: %(default)s)"
    )
    
    parser.add_argument(
        "--no-mol-id",
        action="store_true",
        help="Exclude mol_id column from output"
    )
    
    parser.add_argument(
        "--no-smiles",
        action="store_true",
        help="Exclude smiles column from output"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()

    # Configure logging
    logger.configure(
        handlers=[
            {"sink": sys.stderr, "level": "DEBUG" if args.verbose else "INFO"}
        ]
    )
    
    # Setup paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Determine column inclusion
    include_mol_id = not args.no_mol_id
    include_smiles = not args.no_smiles
    
    try:
        process_molecules(
            input_path,
            output_path,
            include_mol_id=include_mol_id,
            include_smiles=include_smiles,
            verbose=args.verbose
        )
        logger.success(f"Properties saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
