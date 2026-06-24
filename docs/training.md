# Training

The main entrypoint is `scripts/run_training.py`. The `examples/train/` directory contains two self-contained wrappers:

- `pretrain_base.sh` for base-model pretraining
- `sft_universal_multi.sh` for universal multi-condition SFT

Both wrappers accept environment-variable overrides and use `/path/to/` placeholders for datasets. Set `DRY_RUN=1` to print the constructed command without launching training.

## Base Pretraining

```bash
DATA_PATH=/path/to/pretrain.fragment_selfies \
OUTPUT_DIR=/path/to/output/pretrain_base_10m_256h_16l \
NUM_PROCESSES=1 \
examples/train/pretrain_base.sh
```

Equivalent command:

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
  scripts/run_training.py --task pretrain \
  --config_path models/configs/base/config_10m_256h_16l.json \
  --tokenizer_path models/tokenizer \
  --train_data_path /path/to/pretrain.fragment_selfies \
  --output_dir /path/to/output/pretrain_base_10m_256h_16l \
  --batch_size 32 \
  --epochs 2 \
  --learning_rate 2e-4 \
  --max_sequence_length 256
```

## Universal Multi-Condition SFT

```bash
BASE_MODEL_PATH=/path/to/pretrain_base_10m_256h_16l/final_model \
MOLECULE_CONTEXT_PATH=/path/to/molecule_context.fragment_selfies \
PROPERTIES_PATH=/path/to/molecule_properties.csv \
PHARMA_FP_PATH=/path/to/gobbi_pharma_fps.npy \
SAIR_INDEX_DIR=/path/to/SAIR/index \
SAIR_STRUCTURES_DIR=/path/to/SAIR/structures_processed \
SAIR_LIGAND_FRAGMENT_SELFIES_PATH=/path/to/SAIR/ligands.fragment_selfies \
PLINDER_ROOT=/path/to/PLINDER/2024-06/v2 \
PLINDER_INDEX_DIR=/path/to/PLINDER/2024-06/v2/index \
PLINDER_LIGAND_FRAGMENT_SELFIES_PATH=/path/to/PLINDER/ligands.fragment_selfies \
OUTPUT_DIR=/path/to/output/sft_universal_multi_10m_256h_16l \
NUM_PROCESSES=1 \
examples/train/sft_universal_multi.sh
```

Equivalent command:

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
  scripts/run_training.py --task sft --sft_mode universal_multi \
  --base_model /path/to/pretrain_base_10m_256h_16l/final_model \
  --molecule_context_file /path/to/molecule_context.fragment_selfies \
  --properties_file /path/to/molecule_properties.csv \
  --pharma_fp_file /path/to/gobbi_pharma_fps.npy \
  --sair_index_dir /path/to/SAIR/index \
  --sair_structures_dir /path/to/SAIR/structures_processed \
  --sair_ligand_fragment_selfies_file /path/to/SAIR/ligands.fragment_selfies \
  --plinder_root /path/to/PLINDER/2024-06/v2 \
  --plinder_index_dir /path/to/PLINDER/2024-06/v2/index \
  --plinder_ligand_fragment_selfies_file /path/to/PLINDER/ligands.fragment_selfies \
  --target_context_sources sair,plinder \
  --tokenizer_path models/tokenizer \
  --molecule_ratio 4 \
  --target_context_ratio 1 \
  --batch_size 16 \
  --epochs 5 \
  --learning_rate 2e-4
```

## Multi-GPU

Set these environment variables on either wrapper:

```bash
NUM_PROCESSES=8 \
USE_FSDP=true \
FSDP_STRATEGY=full_shard \
MIXED_PRECISION=bf16 \
examples/train/pretrain_base.sh
```

Use `CONDA_ACTIVATE_SCRIPT`, `CONDA_ENV`, and `ENV_SETUP_SCRIPT` if your environment requires activation or module setup before launching Accelerate.
