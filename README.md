# Molexar (Molecular Exalted Architect)

Molexar is a unified multimodal molecular foundation model for drug design that supports unconditional generation and multi-condition generation, including molecular-property conditioning, molecular-pharmacophore conditioning, protein-sequence conditioning, and protein-pocket conditioning, while also accommodating arbitrary custom conditions.

![Molexar architecture](https://raw.githubusercontent.com/fairydance/Molexar/main/images/molexar_architecture.png)

## Resources

- Official website: https://molexar.com
- Molexar source code: https://github.com/fairydance/Molexar
- Fragment-SELFIES source code: https://github.com/fairydance/Fragment-SELFIES
- Unconditional base model: https://huggingface.co/fairydance/molexar-10m-base
- Universal multi-condition model: https://huggingface.co/fairydance/molexar-10m-omni

## Features

- Fragment-SELFIES molecular representation with BRICS fragment tokens
- Unconditional base-model generation and conditional generation in one model class
- Continuous, discrete, and vector condition encoders
- Gemma2 features including RoPE, grouped-query attention, sliding-window attention, and softcapping
- Full-parameter SFT for multi-condition molecular generation
- Training and inference entrypoints for single-GPU and multi-GPU workflows through Accelerate

## Installation

```bash
git clone https://github.com/fairydance/Molexar.git
cd Molexar

conda create -n molexar python=3.13
conda activate molexar

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[train,data]"

python -c "import fragment_selfies; import molexar; print('Molexar environment ready')"
```

[Fragment-SELFIES](https://github.com/fairydance/Fragment-SELFIES) is required for SMILES conversion and generated molecule decoding. Molexar depends on the published `fragment-selfies` package, so `pip install -e .` installs it automatically from PyPI. Install Molexar in editable mode in every environment that runs training, inference, or auxiliary embedding scripts.

## Repository Layout

```text
Molexar/
├── docs/                     # architecture, data, training, and inference guides
├── examples/train/           # training wrappers and shared shell helpers
├── images/                   # image assets used by documentation
├── models/configs/base/      # config_10m_256h_16l.json
├── models/tokenizer/         # tokenizer.json
├── scripts/                  # training, inference, tokenizer, data, and embedding utilities
└── src/molexar/              # package source
```

Datasets are not included in this repository. Released model files are hosted on Hugging Face at [`fairydance/molexar-10m-base`](https://huggingface.co/fairydance/molexar-10m-base) and [`fairydance/molexar-10m-omni`](https://huggingface.co/fairydance/molexar-10m-omni). Pass local dataset and model paths explicitly through CLI flags or environment variables.

## Released Models

- [`fairydance/molexar-10m-base`](https://huggingface.co/fairydance/molexar-10m-base) - unconditional base model for de novo and fragment-constrained generation
- [`fairydance/molexar-10m-omni`](https://huggingface.co/fairydance/molexar-10m-omni) - universal multi-condition model for molecular-property, pharmacophore, protein-sequence, and protein-pocket conditioning

## Base Pretraining

```bash
DATA_PATH=/path/to/pretrain.fragment_selfies \
OUTPUT_DIR=/path/to/output/pretrain_base_10m_256h_16l \
examples/train/pretrain_base.sh
```

Equivalent core command:

```bash
python scripts/run_training.py --task pretrain \
  --config_path models/configs/base/config_10m_256h_16l.json \
  --tokenizer_path models/tokenizer \
  --train_data_path /path/to/pretrain.fragment_selfies \
  --output_dir /path/to/output/pretrain_base_10m_256h_16l \
  --batch_size 32 \
  --epochs 2
```

## Multi-Condition SFT

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
examples/train/sft_universal_multi.sh
```

Use `NUM_PROCESSES`, `MIXED_PRECISION`, `USE_FSDP`, and `FSDP_STRATEGY` environment variables to adapt the wrappers to your hardware.

## Inference

```bash
python scripts/run_inference.py --mode base \
  --model_path /path/to/model/final_model \
  --tokenizer_path models/tokenizer \
  --num_samples 10 \
  --convert_to_smiles \
  --canonical \
  --output_file /path/to/output/base_samples.jsonl \
  --output_format jsonl
```

Conditional generation accepts JSON, NPY, PKL, direct scalar flags, reference SMILES for molecular properties or pharmacophore fingerprints, protein sequences for ESM embeddings, and pocket PDB files for GVP conditioning. See `docs/inference.md` for examples.

## Documentation

- `docs/architecture.md` - model architecture and condition injection
- `docs/data_preparation.md` - expected input file formats
- `docs/training.md` - pretraining and SFT commands
- `docs/inference.md` - base and conditional generation examples

## Citation

```bibtex
@misc{lin2026molexarunifiedmultimodalmolecular,
      title={Molexar: A Unified Multimodal Molecular Foundation Model for Drug Design},
      author={Haoyu Lin and Yiyan Liao and Jinmei Pan and Xinliao Ling and Luhua Lai and Jianfeng Pei},
      year={2026},
      eprint={2606.25865},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2606.25865},
}
```

## License

Molexar is released under the MIT License. See `LICENSE` for details.
