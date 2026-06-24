#!/usr/bin/env python3
"""
Preprocess PLINDER pocket data for 3D-conditioned training.

Batch processes pocket_noH.pdb files to generate:
1. GVP embeddings for each pocket
2. Training dataset JSON with system_id, fragment_selfies, and pocket embeddings
3. Optional: Pre-computed embeddings for faster training

Usage:
    python preprocess_pocket_data.py --csv_path <path> --data_root <path> --output_dir <path>
"""

import sys
import argparse
import pathlib
import json
from loguru import logger
from tqdm import tqdm

try:
    import torch
    import pandas as pd
    from molexar.datasets import PocketProcessor, PLINDERProteinPocketConditionDataset
    from molexar.modeling import MolexarConfig, MolexarModel
except ImportError as e:
    logger.error(f"Required module not found: {e}")
    logger.error("Please install Molexar with pip install -e .")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess PLINDER pocket data for 3D-conditioned training"
    )
    
    # Input paths
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to CSV with system_id and fragment_selfies columns"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing systems_processed subdirectory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/pocket_data",
        help="Directory to save processed data (default: %(default)s)"
    )
    
    # Processing options
    parser.add_argument(
        "--pocket_radius",
        type=float,
        default=10.0,
        help="Pocket radius in Angstroms (default: 10.0)"
    )
    parser.add_argument(
        "--max_atoms",
        type=int,
        default=200,
        help="Maximum atoms per pocket (default: 200)"
    )
    parser.add_argument(
        "--centroid",
        type=str,
        default=None,
        help="Fixed centroid coordinates as 'x,y,z' (default: use file centroid)"
    )
    
    # Model options (for embedding generation)
    parser.add_argument(
        "--generate_embeddings",
        action="store_true",
        help="Generate GVP embeddings using model"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="models/configs/base/config_10m_256h_16l.json",
        help="Path to model config JSON (for embedding generation)"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        help="Path to model checkpoint (for embedding generation)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for embeddings (default: auto)"
    )
    
    # Output format
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "jsonl"],
        default="json",
        help="Output format (default: json)"
    )
    
    return parser.parse_args()


def process_single_system(processor, system_id, data_root, pocket_file="pocket_noH.pdb"):
    try:
        system_dir = pathlib.Path(data_root) / "systems_processed" / system_id
        pocket_path = system_dir / pocket_file
        
        if not pocket_path.exists():
            return None, f"Pocket file not found: {pocket_path}"
        
        pocket_data = processor.process_pdb(str(pocket_path))
        
        data = {
            'system_id': system_id,
            'pocket_data': {
                'node_scalar': pocket_data['node_scalar'].tolist(),
                'node_vector': pocket_data['node_vector'].tolist(),
                'edge_index': pocket_data['edge_index'].tolist(),
                'edge_scalar': pocket_data['edge_scalar'].tolist(),
                'edge_vector': pocket_data['edge_vector'].tolist(),
                'n_atoms': pocket_data['n_atoms'],
                'center': pocket_data['center'].tolist(),
            }
        }
        
        return data, None
        
    except Exception as e:
        return None, str(e)


def main():
    args = parse_args()
    
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    handlers = [
        {"sink": sys.stderr, "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"}
    ]
    handlers.append({"sink": output_dir / "preprocess.log", "rotation": "10 MB", "level": "INFO"})

    logger.configure(handlers=handlers)
    
    logger.info(f"Starting preprocessing with args: {vars(args)}")
    
    logger.info(f"Loading CSV: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    
    required_cols = ['system_id', 'fragment_selfies']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        sys.exit(1)
    
    logger.info(f"Found {len(df)} systems to process")
    
    centroid = None
    if args.centroid:
        coords = tuple(map(float, args.centroid.split(',')))
        centroid = (coords[0], coords[1], coords[2])
    
    processor = PocketProcessor(
        pocket_radius=args.pocket_radius,
        max_atoms=args.max_atoms,
        centroid=centroid
    )
    
    results = []
    errors = []
    
    logger.info("Processing pocket files...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        system_id = row['system_id']
        fragment_selfies = row['fragment_selfies']
        
        data, error = process_single_system(processor, system_id, args.data_root)
        
        if error:
            errors.append({'system_id': system_id, 'error': error})
            logger.warning(f"Failed {system_id}: {error}")
            continue
        
        if data:
            data['fragment_selfies'] = fragment_selfies
            results.append(data)
    
    logger.info(f"Successfully processed {len(results)}/{len(df)} systems")
    
    if args.format == "json":
        output_path = output_dir / "training_data.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved to: {output_path}")
    
    elif args.format == "jsonl":
        output_path = output_dir / "training_data.jsonl"
        with open(output_path, 'w') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Saved to: {output_path}")
    
    if errors:
        error_path = output_dir / "errors.json"
        with open(error_path, 'w') as f:
            json.dump(errors, f, indent=2)
        logger.warning(f"Saved {len(errors)} errors to: {error_path}")
    
    if args.generate_embeddings:
        logger.info("Generating GVP embeddings...")
        
        if not args.model_config or not args.model_checkpoint:
            logger.error("Model config and checkpoint required for embedding generation")
            sys.exit(1)
        
        logger.info(f"Loading model from {args.model_checkpoint}")
        config = MolexarConfig.from_json_file(args.model_config)
        model = MolexarModel(config)
        model.load_state_dict(torch.load(args.model_checkpoint, map_location='cpu'))
        model.to(args.device)
        model.eval()
        
        embeddings = {}
        with torch.no_grad():
            for item in tqdm(results):
                system_id = item['system_id']
                pocket_data = item['pocket_data']
                
                node_scalar = torch.tensor(pocket_data['node_scalar'])
                node_vector = torch.tensor(pocket_data['node_vector'])
                edge_index = torch.tensor(pocket_data['edge_index'])
                edge_scalar = torch.tensor(pocket_data['edge_scalar'])
                edge_vector = torch.tensor(pocket_data['edge_vector'])
                
                node_scalar = node_scalar.to(args.device)
                node_vector = node_vector.to(args.device)
                edge_index = edge_index.to(args.device)
                edge_scalar = edge_scalar.to(args.device)
                edge_vector = edge_vector.to(args.device)
                
                embedding = model.encode_pocket_geometry(
                    node_features=(node_scalar, node_vector),
                    edge_index=edge_index,
                    edge_features=(edge_scalar, edge_vector)
                )
                
                embeddings[system_id] = {
                    'embedding': embedding.cpu().tolist(),
                    'fragment_selfies': item['fragment_selfies']
                }
        
        emb_path = output_dir / "pocket_embeddings.json"
        with open(emb_path, 'w') as f:
            json.dump(embeddings, f, indent=2)
        logger.info(f"Saved embeddings to: {emb_path}")
    
    logger.success("Preprocessing complete!")


if __name__ == "__main__":
    main()
