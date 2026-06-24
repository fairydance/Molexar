# Inference

The main entrypoint is `scripts/run_inference.py`.

## Base Generation

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

## Molecular Property Conditioning

```bash
python scripts/run_inference.py --mode conditional \
  --model_path /path/to/model/final_model \
  --tokenizer_path models/tokenizer \
  --mol_logp 2.5 \
  --mol_tpsa 75.0 \
  --mol_qed 0.7 \
  --num_samples 10 \
  --convert_to_smiles \
  --canonical
```

You can also derive property conditions from a reference SMILES:

```bash
python scripts/run_inference.py --mode conditional \
  --model_path /path/to/model/final_model \
  --reference_smiles "Cc1nnc(N2CCNCC2)s1" \
  --property_keys mol_logp,mol_tpsa,mol_qed \
  --num_samples 10 \
  --convert_to_smiles \
  --canonical
```

## Pharmacophore Fingerprint Conditioning

```bash
python scripts/run_inference.py --mode conditional \
  --model_path /path/to/model/final_model \
  --condition_key mol_pharma_fp \
  --reference_smiles "Cc1nnc(N2CCNCC2)s1" \
  --num_samples 10 \
  --convert_to_smiles \
  --canonical
```

## Protein Sequence Conditioning

```bash
python scripts/run_inference.py --mode conditional \
  --model_path /path/to/model/final_model \
  --protein_sequence "MKTIIALSYIFCLVFAKDRTEG" \
  --num_samples 10 \
  --convert_to_smiles \
  --canonical
```

When `--protein_sequence` or `--protein_sequence_file` is used, Molexar computes an ESM embedding through `scripts/embeddings/compute_esm_embedding.py`. Use `--esm_conda_env`, `--esm_device`, or `--esm_embedding_script` to customize that step.

## Protein Pocket Conditioning

```bash
python scripts/run_inference.py --mode conditional \
  --model_path /path/to/model/final_model \
  --pocket_pdb /path/to/pocket.pdb \
  --pocket_radius 25 \
  --max_atoms 425 \
  --num_samples 10 \
  --convert_to_smiles \
  --canonical
```

## Fragment-Constrained Generation

```bash
python scripts/run_inference.py --mode base \
  --model_path /path/to/model/final_model \
  --generation_task motif_extension \
  --start_smiles '[*]C1(CC#N)CN(S(=O)(=O)CC)C1' \
  --num_samples 10 \
  --convert_to_smiles \
  --canonical
```

Supported tasks are `de_novo`, `motif_extension`, `scaffold_decoration`, `linker_design`, `scaffold_morphing`, and `superstructure`.

## Output Formats

Use `--output_format json`, `jsonl`, `csv`, `tsv`, or `smi`. You can also write separate generated Fragment-SELFIES and SMILES text files:

```bash
python scripts/run_inference.py --mode base \
  --model_path /path/to/model/final_model \
  --num_samples 100 \
  --convert_to_smiles \
  --fragment_selfies_file /path/to/output/generated.fragment_selfies \
  --smiles_file /path/to/output/generated.smi
```
