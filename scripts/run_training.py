#!/usr/bin/env python3
"""Unified Molexar training script for pretraining and SFT."""

import argparse
import json
import os
import sys
from datetime import datetime

from loguru import logger
from transformers import Trainer

from molexar.modeling import MolexarConfig
from molexar.training import (
    ConditionalDataCollator,
    ConditionalTrainer,
    MetricsLoggerCallback,
    build_datasets,
    build_training_arguments,
    find_latest_checkpoint,
    load_model,
    load_tokenizer,
    repair_optimizer_checkpoint,
    resolve_output_paths,
    setup_logging,
)


DEFAULT_DATA_ROOT = "/path/to/datasets"
DEFAULT_TARGET_DATA_ROOT = f"{DEFAULT_DATA_ROOT}/ChEMBL/UniChem"
UNICHEM_WITH_TARGET_PREFIX = (
    f"{DEFAULT_TARGET_DATA_ROOT}/"
    "unichem_canonical_smiles.no_hydrogen.no_stereo.deduplicated.filtered.with_sair_and_plinder"
)
UNICHEM_TARGET_SMILES = f"{UNICHEM_WITH_TARGET_PREFIX}.smi"
UNICHEM_TARGET_RANDOMIZED_FRAGMENT_SELFIES = f"{UNICHEM_WITH_TARGET_PREFIX}.randomized.4_fold.fragment_selfies"
UNICHEM_TARGET_PROPERTIES = f"{UNICHEM_WITH_TARGET_PREFIX}.properties.csv"
UNICHEM_TARGET_PHARMA_FP = f"{UNICHEM_WITH_TARGET_PREFIX}.gobbi_pharma_fps.npy"
UNICHEM_RANDOMIZED_FRAGMENT_SELFIES = UNICHEM_TARGET_RANDOMIZED_FRAGMENT_SELFIES
UNICHEM_SMILES = UNICHEM_TARGET_SMILES
SAIR_ROOT = "/path/to/SAIR"
SAIR_INDEX_DIR = f"{SAIR_ROOT}/index"
SAIR_STRUCTURES_DIR = f"{SAIR_ROOT}/structures_processed"
PLINDER_ROOT = "/path/to/PLINDER/2024-06/v2"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Molexar training for pretraining and conditional SFT."
    )
    parser.add_argument("--task", choices=["pretrain", "sft"], default="pretrain")
    parser.add_argument(
        "--sft_mode",
        choices=[
            "generic",
            "properties",
            "pharma_fp",
            "molecule_multi",
            "protein_sequence",
            "protein_pocket",
            "universal_multi",
        ],
        default="generic",
    )

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--final_output_dir", type=str, default=None)
    parser.add_argument("--metrics_dir", type=str, default=None)
    parser.add_argument("--tokenizer_output_dir", type=str, default=None)
    parser.add_argument("--log_file", type=str, default=None)

    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default="models/tokenizer")

    parser.add_argument(
        "--train_data_path",
        type=str,
        default=UNICHEM_RANDOMIZED_FRAGMENT_SELFIES,
    )
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--sft_data_path", type=str, default=None)
    parser.add_argument("--val_sft_data_path", type=str, default=None)

    parser.add_argument(
        "--smiles_file",
        type=str,
        default=UNICHEM_SMILES,
    )
    parser.add_argument(
        "--molecule_context_file",
        type=str,
        default=UNICHEM_TARGET_RANDOMIZED_FRAGMENT_SELFIES,
        help="Randomized Fragment-SELFIES file for universal_multi molecule-context SFT",
    )
    parser.add_argument("--val_smiles_file", type=str, default=None)
    parser.add_argument(
        "--properties_file",
        type=str,
        default=UNICHEM_TARGET_PROPERTIES,
    )
    parser.add_argument("--val_properties_file", type=str, default=None)
    parser.add_argument(
        "--pharma_fp_file",
        type=str,
        default=UNICHEM_TARGET_PHARMA_FP,
    )
    parser.add_argument("--val_pharma_fp_file", type=str, default=None)

    parser.add_argument(
        "--plinder_root",
        type=str,
        default=PLINDER_ROOT,
    )
    parser.add_argument("--plinder_index_dir", type=str, default=f"{PLINDER_ROOT}/index")
    parser.add_argument("--sair_index_dir", type=str, default=SAIR_INDEX_DIR)
    parser.add_argument("--sair_structures_dir", type=str, default=SAIR_STRUCTURES_DIR)
    parser.add_argument(
        "--sair_pairs_file",
        type=str,
        default=None,
        help="SAIR target-context protein-ligand pair CSV override",
    )
    parser.add_argument(
        "--plinder_pairs_file",
        type=str,
        default=None,
        help="PLINDER target-context protein-ligand pair CSV override",
    )
    parser.add_argument(
        "--target_context_sources",
        type=str,
        default="sair,plinder",
        help="Comma-separated target-context sources: sair, plinder",
    )
    parser.add_argument(
        "--target_context_cache_dir",
        type=str,
        default=None,
        help="Directory for cached target-context metadata",
    )
    parser.add_argument(
        "--verify_target_pocket_paths",
        action="store_true",
        default=True,
        help="Verify SAIR/PLINDER pocket files during target-context metadata loading",
    )
    parser.add_argument(
        "--no_verify_target_pocket_paths",
        action="store_false",
        dest="verify_target_pocket_paths",
        help="Disable target pocket path verification during metadata loading",
    )
    parser.add_argument("--pocket_radius", type=float, default=25.0)
    parser.add_argument("--max_atoms", type=int, default=425)

    parser.add_argument("--data_format", choices=["fragment_selfies", "smiles"], default="fragment_selfies")
    parser.add_argument("--fallback_selfies", action="store_true", default=False)
    conversion = parser.add_mutually_exclusive_group()
    conversion.add_argument("--canonical", action="store_true")
    conversion.add_argument("--randomized", action="store_true")
    parser.add_argument("--slicer", type=str, default="brics")
    parser.add_argument(
        "--implicit_probability",
        "--implicit-probability",
        dest="implicit_probability",
        type=float,
        default=None,
        help="Probability that randomized auto-style SMILES conversion emits implicit adjacent-root anchors",
    )
    parser.add_argument(
        "--max_implicit_cuts",
        type=int,
        default=None,
        help="Maximum implicit BRICS edges to cut per connected component",
    )
    parser.add_argument(
        "--fragment_style",
        choices=["auto", "explicit", "implicit"],
        default=None,
        help="Fragment-SELFIES style for on-the-fly SMILES conversion",
    )
    parser.add_argument("--require-hs", action="store_true", default=False)
    parser.add_argument("--no-require-hs", action="store_false", dest="require_hs")
    parser.add_argument("--ignore-stereo", action="store_true", default=True)
    parser.add_argument("--no-ignore-stereo", action="store_false", dest="ignore_stereo")

    parser.add_argument("--disable_random_condition_subset", action="store_true", default=False)
    parser.add_argument("--min_condition_count", type=int, default=1)
    parser.add_argument("--max_condition_count", type=int, default=None)
    parser.add_argument("--single_condition_weight", type=float, default=0.6)
    parser.add_argument("--dual_condition_weight", type=float, default=0.3)
    parser.add_argument("--triple_condition_weight", type=float, default=0.1)
    parser.add_argument("--molecule_pharma_fp_condition_probability", type=float, default=0.5)
    parser.add_argument("--target_dual_context_probability", type=float, default=0.2)
    parser.add_argument("--target_ligand_condition_probability", type=float, default=0.2)
    parser.add_argument("--target_ligand_fragment_selfies_folds", type=int, default=10)
    parser.add_argument("--sair_ligand_fragment_selfies_file", default=None)
    parser.add_argument("--plinder_ligand_fragment_selfies_file", default=None)
    parser.add_argument("--skip_fragment_condition_alignment_check", action="store_true", default=False)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--molecule_ratio", type=int, default=9)
    parser.add_argument("--target_context_ratio", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--eval_steps", type=int, default=500)

    parser.add_argument("--use_fsdp", action="store_true")
    parser.add_argument(
        "--fsdp_strategy",
        choices=["full_shard", "shard_grad_op", "no_shard", "hybrid_shard", "hybrid_shard_zero2"],
        default="full_shard",
    )
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.provided_args = {item.split("=", 1)[0] for item in sys.argv[1:] if item.startswith("--")}
    return args


def apply_default_sft_paths(args):
    if args.task != "sft":
        return

    if args.sft_mode == "molecule_multi" and "--smiles_file" not in args.provided_args:
        args.smiles_file = UNICHEM_RANDOMIZED_FRAGMENT_SELFIES

    uses_target_context = args.sft_mode in {"protein_sequence", "protein_pocket", "universal_multi"}

    if args.sft_mode == "universal_multi":
        if "--molecule_context_file" not in args.provided_args:
            args.molecule_context_file = UNICHEM_TARGET_RANDOMIZED_FRAGMENT_SELFIES
        if "--smiles_file" not in args.provided_args:
            args.smiles_file = UNICHEM_TARGET_SMILES
        if "--properties_file" not in args.provided_args:
            args.properties_file = UNICHEM_TARGET_PROPERTIES
        if "--pharma_fp_file" not in args.provided_args:
            args.pharma_fp_file = UNICHEM_TARGET_PHARMA_FP
        if "--canonical" not in args.provided_args and "--randomized" not in args.provided_args:
            args.randomized = True
        if "--implicit_probability" not in args.provided_args and "--implicit-probability" not in args.provided_args:
            args.implicit_probability = 0.15

    if uses_target_context and "--plinder_index_dir" not in args.provided_args:
        args.plinder_index_dir = os.path.join(args.plinder_root, "index")


def main():
    args = parse_args()
    if min(args.molecule_ratio, args.target_context_ratio) < 0:
        raise ValueError("universal_multi ratios must be non-negative")
    if args.implicit_probability is not None and not 0.0 <= args.implicit_probability <= 1.0:
        raise ValueError("--implicit_probability must be between 0 and 1")
    condition_weight_sum = args.single_condition_weight + args.dual_condition_weight + args.triple_condition_weight
    if condition_weight_sum <= 0:
        raise ValueError("At least one condition-count weight must be positive")
    if min(args.single_condition_weight, args.dual_condition_weight, args.triple_condition_weight) < 0:
        raise ValueError("Condition-count weights must be non-negative")
    if not 0.0 <= args.molecule_pharma_fp_condition_probability <= 1.0:
        raise ValueError("--molecule_pharma_fp_condition_probability must be between 0 and 1")
    if not 0.0 <= args.target_dual_context_probability <= 1.0:
        raise ValueError("--target_dual_context_probability must be between 0 and 1")
    if not 0.0 <= args.target_ligand_condition_probability <= 1.0:
        raise ValueError("--target_ligand_condition_probability must be between 0 and 1")
    if args.target_ligand_fragment_selfies_folds < 1:
        raise ValueError("--target_ligand_fragment_selfies_folds must be positive")
    if args.max_steps < -1 or args.max_steps == 0:
        raise ValueError("--max_steps must be a positive integer, or -1 to train by epochs")
    apply_default_sft_paths(args)
    if args.seed is not None:
        import random
        import numpy as np
        import torch

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    default_name = "pretrain" if args.task == "pretrain" else f"sft_{args.sft_mode}"
    paths = resolve_output_paths(args, default_name)
    setup_logging(paths.log_file, args.verbose)

    if args.config_path:
        config = MolexarConfig.from_pretrained(args.config_path)
    elif args.task == "sft" and args.base_model:
        config = MolexarConfig.from_pretrained(args.base_model)
    else:
        config = MolexarConfig()
    if args.task == "sft" and args.sft_mode in {"protein_pocket", "universal_multi"}:
        node_scalar_dim, node_vector_dim = config.gvp_node_in_dim
        if node_scalar_dim == 6:
            config.gvp_node_in_dim = (11, node_vector_dim)
            logger.info("Using 11 scalar pocket atom features for GVP input")
    config.max_position_embeddings = args.max_sequence_length
    config.sliding_window = args.max_sequence_length // 2

    tokenizer = load_tokenizer(args, paths, config)
    config.vocab_size = len(tokenizer)
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id

    model = load_model(args, config)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    logger.info(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}")

    train_dataset, eval_dataset, conditional = build_datasets(args, tokenizer, config)
    trainer_cls = ConditionalTrainer if conditional else Trainer
    data_collator = ConditionalDataCollator() if conditional else None
    training_args = build_training_arguments(args, paths, eval_dataset is not None)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "callbacks": [MetricsLoggerCallback(paths.metrics_dir)],
    }
    if data_collator is not None:
        trainer_kwargs["data_collator"] = data_collator
    trainer = trainer_cls(**trainer_kwargs)

    resume = None if args.no_resume else find_latest_checkpoint(paths.checkpoint_dir)
    if resume and trainer.is_world_process_zero():
        repair_optimizer_checkpoint(resume)
    if resume:
        trainer.accelerator.wait_for_everyone()
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(paths.final_output_dir)
    tokenizer.save_pretrained(paths.final_output_dir)
    config.save_pretrained(paths.final_output_dir)

    summary = {
        "task": args.task,
        "sft_mode": args.sft_mode if args.task == "sft" else None,
        "timestamp": datetime.now().isoformat(),
        "model_params": total_params,
        "trainable_params": trainable_params,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
    }
    with open(os.path.join(paths.metrics_dir, "training_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    logger.success(f"Training complete. Final model saved to {paths.final_output_dir}")


if __name__ == "__main__":
    main()
