# Architecture

Molexar uses a Gemma2-style causal decoder for Fragment-SELFIES molecular generation. The included configuration is `models/configs/base/config_10m_256h_16l.json`, a 16-layer, 256-hidden model with sliding-window attention and a 256-token context window.

## Sequence Format

Unconditional examples use:

```text
<BOS><MOL>{fragment_selfies}</MOL><EOS>
```

Conditional examples use:

```text
<BOS>[conditions]<SEP><MOL>{fragment_selfies}</MOL><EOS>
```

The model inserts condition embeddings in `condition_settings` order and adds a single `<SEP>` token before the molecular span whenever conditions are present.

## Condition Families

- Discrete molecule properties: `mol_hac`, `mol_hbdc`, `mol_hbac`, `mol_rotbc`
- Continuous molecule properties: `mol_wt`, `mol_logp`, `mol_tpsa`, `mol_qed`, `mol_sas`
- Vector conditions: `mol_pharma_fp`, `prot_seq_esm_emb`, `prot_poc_gvp_emb`

Discrete values use one-hot encoders. Continuous values use radial basis functions followed by projection layers. Vector conditions are projected directly into the model hidden size.

## Pocket Conditioning

Protein-pocket conditioning uses a GVP encoder over pocket atom graphs. Pocket PDB inputs are converted into node and edge features at inference time. During universal multi-condition SFT, target-context datasets provide indexed protein-pocket records plus aligned ligand Fragment-SELFIES files.

## Generation Modes

- Base generation starts from `<BOS><MOL>` and samples Fragment-SELFIES tokens.
- Conditional generation prepends one or more condition embeddings, then samples the molecule span.
- Fragment-constrained generation can seed generation from a Fragment-SELFIES prefix or a SMILES-derived task constraint.
