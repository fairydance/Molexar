# Molexar Scripts

Primary entrypoints stay at the top level:

- `run_training.py` - unified pretraining and SFT training
- `run_inference.py` - unified base and conditional generation; see `../docs/inference.md` for output formats and sampling commands

See `../docs/training.md` for pretraining and universal SFT commands.

Utility scripts are grouped by purpose:

- `data/` - conversion, SFT preparation, start-fragment extraction, pocket preprocessing, and molecular property calculation utilities
- `tokenizer/` - tokenizer training and Fragment-SELFIES tokenization utilities
- `embeddings/` - external condition embedding generation

Install Molexar in editable mode before running scripts:

```bash
python -m pip install -e /path/to/Molexar --no-deps
python scripts/run_training.py --help
python scripts/run_inference.py --help
```
