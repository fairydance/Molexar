# Model Configuration

This directory contains the default Molexar base model configuration.

## Config Naming Convention

All configuration files follow a standardized naming pattern:

```
config_{params}m_{hidden}h_{layers}l.json
```

**Components:**
- **`{params}`**: Approximate transformer backbone parameter count in millions (e.g., `10m` = ~10M backbone parameters)
- **`{hidden}`**: Hidden size dimension (e.g., `1024h` = 1024 hidden units)
- **`{layers}`**: Number of transformer layers (e.g., `20l` = 20 layers)

**Included config:**
- `config_10m_256h_16l.json` - 10.53M transformer backbone parameters, 256 hidden size, 16 layers

**Key Config Parameters:**
- `bos_token_id`, `eos_token_id`, `pad_token_id`: 1, 2, 0 (Fragment-SELFIES format)

**Location:** `models/configs/base/config_10m_256h_16l.json`

Condition encoders are always present in the unified model config. Pre-training freezes those
modules, while SFT loads base weights selectively and trains condition encoders from scratch.

## 10M Unified Model Parameter Breakdown

`models/configs/base/config_10m_256h_16l.json` is the default Molexar config. It uses:

- `hidden_size`: 256
- `intermediate_size`: 640
- `num_hidden_layers`: 16
- `num_attention_heads`: 4
- `num_key_value_heads`: 1
- `head_dim`: 64
- `condition_projector_layers`: 2

Current total parameters: **14,742,693**.

| Component | Params |
|-----------|--------|
| Transformer layers | 10,502,144 |
| Token embedding | 32,512 |
| Final norm | 256 |
| Condition encoders | 1,546,240 |
| GVP encoder | 2,661,541 |
| Separate LM head | 0 |

The LM head is tied to token embeddings, so it does not add separate parameters.

### Transformer Layer Breakdown

Each transformer layer has 656,384 parameters.

| Component | Params |
|-----------|--------|
| Self-attention total | 2,621,440 |
| MLP total | 7,864,320 |
| Layer norms total | 16,384 |

### Condition Encoder Breakdown

| Condition | Params |
|-----------|--------|
| `mol_hac` | 78,592 |
| `mol_hbdc` | 68,864 |
| `mol_hbac` | 71,936 |
| `mol_rotbc` | 71,424 |
| `mol_wt` | 86,528 |
| `mol_logp` | 94,720 |
| `mol_tpsa` | 90,624 |
| `mol_qed` | 78,336 |
| `mol_sas` | 82,432 |
| `mol_pharma_fp` | 330,240 |
| `prot_seq_esm_emb` | 360,960 |
| `prot_poc_gvp_emb` | 131,584 |

### GVP Encoder Breakdown

| GVP Part | Params |
|----------|--------|
| `W_v` | 6,704 |
| `W_e` | 1,154 |
| `layers` | 2,583,027 |
| `W_out` | 70,656 |

## Usage

### Training with the included config:
```bash
python scripts/run_training.py --task pretrain \
  --config_path models/configs/base/config_10m_256h_16l.json \
  --train_data_path /path/to/pretrain.fragment_selfies \
  --batch_size 16 \
  --epochs 2
```

### Creating a custom config:
Copy an existing config and modify parameters:
```json
{
  "vocab_size": 127,
  "hidden_size": 1024,
  "intermediate_size": 2560,
  "num_hidden_layers": 20,
  "num_attention_heads": 8,
  "num_key_value_heads": 4,
  "max_position_embeddings": 256,
  "sliding_window": 128,
  "attention_bias": false,
  "attention_dropout": 0.0,
  "rms_norm_eps": 1e-06,
  "hidden_activation": "gelu_pytorch_tanh",
  "initializer_range": 0.02
}
```

## Parameter Calculation

The total parameters are approximately:
```
vocab_size * hidden_size +  # Embeddings
4 * hidden_size^2 * num_layers +  # Attention
2 * hidden_size * intermediate_size * num_layers +  # MLP
4 * hidden_size * num_layers  # Layer norms
```

The included config uses:
- vocab_size: 127 tokens for the current trained Fragment-SELFIES tokenizer; training scripts overwrite it with the loaded tokenizer size
- max_position_embeddings: 256 total tokens
- sliding_window: 128 tokens
- attention_bias: false
- rms_norm_eps: 1e-06
- hidden_activation: gelu_pytorch_tanh
- initializer_range: 0.02
