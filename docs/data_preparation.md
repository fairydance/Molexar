# Data Preparation

Molexar expects text and array files to be supplied by path at runtime. Datasets are not bundled with the repository.

## Pretraining Corpus

Use one Fragment-SELFIES molecule per line:

```text
[Frag][C][C][O]
[Frag][c][Branch1][C][N]
```

If a line contains extra columns after whitespace, only the first field is used by tokenizer training and data loading.

SMILES files can also be used with on-the-fly conversion:

```bash
python scripts/run_training.py --task pretrain \
  --data_format smiles \
  --randomized \
  --slicer brics \
  --train_data_path /path/to/pretrain.smi \
  --output_dir /path/to/output/pretrain_base
```

## Molecular Properties

Property CSV files should be line-aligned to the molecule context file and include these columns:

```text
hac,hbdc,hbac,rotbc,wt,logp,tpsa,qed,sas
```

## Pharmacophore Fingerprints

Pharmacophore fingerprints are loaded from `.npy`, JSON, or PKL files depending on the CLI entrypoint. For universal multi-condition SFT, use a `.npy` array aligned to the molecule context file.

## Target-Context Data

Universal multi-condition SFT can combine molecule-context rows with SAIR and PLINDER-style target-context records. Provide these paths explicitly:

- `--sair_index_dir`
- `--sair_structures_dir`
- `--sair_ligand_fragment_selfies_file`
- `--plinder_root`
- `--plinder_index_dir`
- `--plinder_ligand_fragment_selfies_file`

Ligand Fragment-SELFIES files should contain one randomized ligand representation per line. Use `--target_ligand_fragment_selfies_folds` when each target ligand has multiple randomized views.

## Tokenizer Training

```bash
python scripts/tokenizer/train_tokenizer.py \
  --data_file /path/to/pretrain.fragment_selfies \
  --output_dir models/tokenizer
```

The repository includes `models/tokenizer/tokenizer.json`, so retraining is optional unless you change the molecular vocabulary.
