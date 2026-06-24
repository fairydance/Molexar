#!/usr/bin/env python3
"""
ESM Embedding Computation for Protein Sequences

Computes ESM embeddings for a single protein amino acid sequence.
The script must be run in the molexar_aux environment where esm is installed.

Usage:
    # Activate the molexar_aux environment first
    conda activate molexar_aux
    
    # Run the script
    python scripts/embeddings/compute_esm_embedding.py --sequence "MKTIIALSYIFCLVFAKDRTEG" --output output.json
    
    # Or with input file
    python scripts/embeddings/compute_esm_embedding.py --input protein.txt --output output.pkl --format pkl --storage numpy
    
    # Save as PKL with Python list (default is numpy array)
    python scripts/embeddings/compute_esm_embedding.py --sequence "MKTIIALSYIFCLVFAKDRTEG" --output output.pkl --format pkl --storage list
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

from loguru import logger


ESMC_MODEL_NAME = "esmc_600m"
ESMC_EMBEDDING_DIM = 1152


def load_sequence_from_file(input_path: Path) -> str:
    """
    Load a protein sequence from a text file.
    
    Args:
        input_path: Path to the input file
        
    Returns:
        Protein sequence as a string
    """
    logger.info(f"Loading sequence from: {input_path}")
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"File not found: {input_path}")
    
    with open(input_path, 'r') as f:
        content = f.read().strip()
    
    # Remove any whitespace and newlines
    sequence = ''.join(content.split())
    
    if not sequence:
        logger.error("Empty sequence in input file")
        raise ValueError("Empty sequence")
    
    logger.info(f"Loaded sequence of length: {len(sequence)}")
    return sequence


def compute_embedding(sequence: str, device: str = "cuda") -> Any:
    """
    Compute ESM embedding for a single protein sequence.
    
    Args:
        sequence: Amino acid sequence
        device: Device to use ('cuda' or 'cpu')

    Returns:
        1D NumPy array of embedding values
    """
    logger.info(f"Computing ESM embedding for sequence of length {len(sequence)}")
    logger.info("Using final ESMC embeddings with residue-only mean pooling")
    
    try:
        # Convert device string to torch.device object
        import torch
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig

        device_obj = torch.device(device)
        
        logger.info(f"Loading ESM model ({ESMC_MODEL_NAME})...")
        client = ESMC.from_pretrained(ESMC_MODEL_NAME, device=device_obj)
        logger.success(f"Model loaded successfully on {device}")

        protein = ESMProtein(sequence=sequence)
        logger.info("Encoding protein...")
        protein_tensor = client.encode(protein)

        logger.info("Computing final embeddings...")
        EMBEDDING_CONFIG = LogitsConfig(return_embeddings=True)
        logits_output = client.logits(protein_tensor, EMBEDDING_CONFIG)
        if logits_output.embeddings is None:
            raise RuntimeError("ESMC did not return embeddings")

        token_embeddings = logits_output.embeddings.float()
        if token_embeddings.dim() != 3 or token_embeddings.shape[0] != 1:
            raise ValueError(f"Unexpected ESMC embedding shape: {tuple(token_embeddings.shape)}")
        if token_embeddings.shape[1] <= 2:
            raise ValueError(
                "Protein sequence produced no residue embeddings after removing special tokens"
            )

        # ESMC tokenization adds <cls> and <eos>; exclude them from the protein representation.
        residue_embeddings = token_embeddings[:, 1:-1, :]
        embedding = residue_embeddings.mean(dim=1).squeeze(0)
        if embedding.shape[0] != ESMC_EMBEDDING_DIM:
            raise ValueError(
                f"Unexpected {ESMC_MODEL_NAME} embedding dimension: "
                f"{embedding.shape[0]} != {ESMC_EMBEDDING_DIM}"
            )

        embedding_1d = embedding.cpu().numpy()

        logger.success(f"Embedding computed successfully: shape {embedding_1d.shape}")
        return embedding_1d
        
    except ImportError as e:
        logger.error(f"ESM library not available: {e}")
        logger.error("Please activate the molexar_aux environment:")
        logger.error("  conda activate molexar_aux")
        logger.error("Then install esm:")
        logger.error("  pip install esm")
        raise
    except Exception as e:
        logger.error(f"Error computing embedding: {e}")
        raise


def save_embedding(
    embedding: Any,
    output_path: Path,
    format: str = "pkl",
    storage: str = "numpy"
) -> None:
    """
    Save embedding to disk in the specified format.
    
    Args:
        embedding: NumPy array of embedding values
        output_path: Path to save the file
        format: Output format ('json' or 'pkl')
        storage: Storage type for PKL ('numpy' or 'list')
    """
    logger.info(f"Saving embedding to {output_path}")
    logger.info(f"Format: {format}, Storage: {storage}")
    
    if format == "json":
        # Convert to list for JSON serialization
        embedding_list = embedding.tolist()
        
        # Create output structure
        output_data = {
            "prot_seq_esm_emb": embedding_list
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.success(f"Saved embedding as JSON: {output_path}")
        
    elif format == "pkl":
        # Prepare data based on storage type
        if storage == "numpy":
            # Store as NumPy array
            output_data = {
                "prot_seq_esm_emb": embedding
            }
        elif storage == "list":
            # Convert to Python list
            output_data = {
                "prot_seq_esm_emb": embedding.tolist()
            }
        else:
            raise ValueError(f"Unknown storage type: {storage}")
        
        # Save as pickle
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.success(f"Saved embedding as PKL: {output_path}")
        
    else:
        raise ValueError(f"Unknown format: {format}")
    
    # Log sample values
    logger.info(f"Embedding shape: {embedding.shape}")
    logger.info(f"Sample values (first 5): {embedding[:5]}")
    logger.info(f"Min: {embedding.min():.6f}, Max: {embedding.max():.6f}")
    logger.info(f"Mean: {embedding.mean():.6f}, Std: {embedding.std():.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute ESM embedding for a protein sequence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute embedding from sequence string (save as JSON)
  python scripts/embeddings/compute_esm_embedding.py --sequence "MKTIIALSYIFCLVFAKDRTEG" --output embedding.json --format json
  
  # Compute embedding from sequence string (save as PKL with numpy array)
  python scripts/embeddings/compute_esm_embedding.py --sequence "MKTIIALSYIFCLVFAKDRTEG" --output embedding.pkl --format pkl --storage numpy
  
  # Compute embedding from sequence string (save as PKL with Python list)
  python scripts/embeddings/compute_esm_embedding.py --sequence "MKTIIALSYIFCLVFAKDRTEG" --output embedding.pkl --format pkl --storage list
  
  # Compute embedding from input file
  python scripts/embeddings/compute_esm_embedding.py --input protein.txt --output embedding.json --format json
  
  # Use CPU instead of GPU
  python scripts/embeddings/compute_esm_embedding.py --sequence "MKTIIALSYIFCLVFAKDRTEG" --output embedding.pkl --device cpu
  
  # Default behavior (PKL with numpy array)
  python scripts/embeddings/compute_esm_embedding.py --sequence "MKTIIALSYIFCLVFAKDRTEG" --output embedding.pkl
        """
    )
    
    parser.add_argument(
        "--sequence",
        type=str,
        help="Protein amino acid sequence (e.g., 'MKTIIALSYIFCLVFAKDRTEG')"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input file containing protein sequence"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="output/esm_embedding.pkl",
        help="Path to save the output file (default: %(default)s)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "pkl"],
        default="pkl",
        help="Output format (default: pkl)"
    )
    
    parser.add_argument(
        "--storage",
        type=str,
        choices=["list", "numpy"],
        default="numpy",
        help="Storage type for PKL format (default: numpy)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use (default: cuda)"
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
    
    # Validate input
    if not args.sequence and not args.input:
        logger.error("Either --sequence or --input must be provided")
        sys.exit(1)
    
    if args.sequence and args.input:
        logger.error("Cannot specify both --sequence and --input")
        sys.exit(1)
    
    # Load sequence
    try:
        if args.input:
            sequence = load_sequence_from_file(Path(args.input))
        else:
            sequence = ''.join(args.sequence.split())
            if not sequence or len(sequence) == 0:
                logger.error("Empty sequence provided")
                sys.exit(1)
            logger.info(f"Using provided sequence of length: {len(sequence)}")
    except Exception as e:
        logger.error(f"Failed to load sequence: {e}")
        sys.exit(1)
    
    # Check device availability
    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"
            else:
                device = "cuda"
        except ImportError:
            logger.warning("PyTorch not available, falling back to CPU")
            device = "cpu"
    else:
        device = "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Compute embedding
    try:
        embedding = compute_embedding(sequence, device)
    except Exception as e:
        logger.error(f"Failed to compute embedding: {e}")
        sys.exit(1)
    
    # Save embedding
    try:
        output_path = Path(args.output)
        save_embedding(embedding, output_path, args.format, args.storage)
    except Exception as e:
        logger.error(f"Failed to save embedding: {e}")
        sys.exit(1)
    
    # Summary
    logger.success("\n" + "="*80)
    logger.success("ESM embedding computation complete!")
    logger.success(f"Sequence length: {len(sequence)}")
    logger.success(f"Embedding shape: {embedding.shape}")
    logger.success(f"Output file: {args.output}")
    logger.success(f"Format: {args.format}")
    if args.format == "pkl":
        logger.success(f"Storage type: {args.storage}")
    logger.success("="*80)


if __name__ == "__main__":
    main()
