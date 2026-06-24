from torch.utils.data import Dataset
from loguru import logger
from molexar.modeling import MolexarConfig


class PretrainingDataset(Dataset):
    """
    Dataset for base model pre-training with full template format.
    
    Template: <BOS><COND><PROP><VALUE>...</COND><SEP><MOL>{molecule}</MOL><EOS>
    """
    
    def __init__(self, file_path, tokenizer, config: MolexarConfig, max_sequence_length=256):
        with open(file_path, "r") as f:
            self.lines = []
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if parts:
                    self.lines.append(parts[0])
        self.tokenizer = tokenizer
        self.config = config
        self.max_sequence_length = max_sequence_length
        
        self._logged_template = False
        self._log_count = 0
        
        self._build_template_parts()
        
        logger.info(f"PretrainingDataset initialized with {len(self.lines)} samples")
        logger.info(f"Template format: <BOS><COND>...<MOL>{{molecule}}</MOL><EOS>")
        self._log_token_info()

    def _build_template_parts(self):
        """Build the condition block parts of the template."""
        c = self.config
        
        self.cond_block = (
            f"{c.COND_TOKEN}"
            f"{c.MOL_HAC_TOKEN}{c.VALUE_TOKEN}"
            f"{c.MOL_HBDC_TOKEN}{c.VALUE_TOKEN}"
            f"{c.MOL_HBAC_TOKEN}{c.VALUE_TOKEN}"
            f"{c.MOL_ROTBC_TOKEN}{c.VALUE_TOKEN}"
            f"{c.MOL_WT_TOKEN}{c.VALUE_TOKEN}"
            f"{c.MOL_LOGP_TOKEN}{c.VALUE_TOKEN}"
            f"{c.MOL_TPSA_TOKEN}{c.VALUE_TOKEN}"
            f"{c.MOL_QED_TOKEN}{c.VALUE_TOKEN}"
            f"{c.MOL_SAS_TOKEN}{c.VALUE_TOKEN}"
            f"{c.MOL_PHARMA_FP_TOKEN}{c.VALUE_TOKEN}"
            f"{c.PROT_SEQ_ESM_EMB_TOKEN}{c.VALUE_TOKEN}"
            f"{c.PROT_POC_GVP_EMB_TOKEN}{c.VALUE_TOKEN}"
            f"{c.UNUSED0_TOKEN}{c.VALUE_TOKEN}"
            f"{c.UNUSED1_TOKEN}{c.VALUE_TOKEN}"
            f"{c.UNUSED2_TOKEN}{c.VALUE_TOKEN}"
            f"{c.UNUSED3_TOKEN}{c.VALUE_TOKEN}"
            f"{c.UNUSED4_TOKEN}{c.VALUE_TOKEN}"
            f"{c.UNUSED5_TOKEN}{c.VALUE_TOKEN}"
            f"{c.UNUSED6_TOKEN}{c.VALUE_TOKEN}"
            f"{c.UNUSED7_TOKEN}{c.VALUE_TOKEN}"
            f"{c.UNUSED8_TOKEN}{c.VALUE_TOKEN}"
            f"{c.UNUSED9_TOKEN}{c.VALUE_TOKEN}"
            f"{c.COND_END_TOKEN}"
        )
        self.mol_prefix = f"{c.SEP_TOKEN}{c.MOL_TOKEN}"
        
        self.mol_suffix = f"{c.MOL_END_TOKEN}{c.EOS_TOKEN}"

    def _log_token_info(self):
        """Log key token information."""
        c = self.config
        tokens_to_check = [
            ('BOS', c.BOS_TOKEN),
            ('COND', c.COND_TOKEN),
            ('COND_END', c.COND_END_TOKEN),
            ('SEP', c.SEP_TOKEN),
            ('MOL', c.MOL_TOKEN),
            ('MOL_END', c.MOL_END_TOKEN),
            ('EOS', c.EOS_TOKEN),
            ('VALUE', c.VALUE_TOKEN),
        ]
        
        logger.debug("Key token IDs:")
        for name, token in tokens_to_check:
            idx = c.special_tokens_list.index(token)
            logger.debug(f"  {name}: '{token}' -> {idx}")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        molecule = self.lines[idx]
        
        formatted = (
            f"{self.config.BOS_TOKEN}"
            f"{self.cond_block}"
            f"{self.mol_prefix}"
            f"{molecule}"
            f"{self.mol_suffix}"
        )
        
        if not self._logged_template and self._log_count < 3:
            self._log_count += 1
            self._log_sample_verification(idx, molecule, formatted)
            if self._log_count >= 3:
                self._logged_template = True
        
        encodings = self.tokenizer(
            formatted, 
            add_special_tokens=False,
            truncation=True, 
            max_length=self.max_sequence_length,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encodings['input_ids'].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": encodings['attention_mask'].squeeze(0),
            "labels": input_ids.clone()
        }
    
    def _log_sample_verification(self, idx, molecule, formatted):
        """Log sample verification details."""
        logger.debug(f"Sample {idx} template verification:")
        logger.debug(f"  Molecule (first 50 chars): {molecule[:50]}...")
        logger.debug(f"  Template length: {len(formatted)} chars")
        
        tokens = self.tokenizer.encode(formatted, add_special_tokens=False)
        token_strs = self.tokenizer.convert_ids_to_tokens(tokens[:30])
        logger.debug(f"  First 30 tokens: {token_strs}")
        logger.debug(f"  First 30 token IDs: {tokens[:30]}")
        logger.debug(f"  Total tokens: {len(tokens)}")
        
        c = self.config
        bos_id = c.special_tokens_list.index(c.BOS_TOKEN)
        cond_id = c.special_tokens_list.index(c.COND_TOKEN)
        cond_end_id = c.special_tokens_list.index(c.COND_END_TOKEN)
        sep_id = c.special_tokens_list.index(c.SEP_TOKEN)
        mol_id = c.special_tokens_list.index(c.MOL_TOKEN)
        
        checks = [
            (0, bos_id, "BOS"),
            (1, cond_id, "COND"),
        ]
        
        all_ok = True
        for pos, expected_id, name in checks:
            actual = tokens[pos] if pos < len(tokens) else -1
            status = "OK" if actual == expected_id else "MISMATCH"
            if actual != expected_id:
                all_ok = False
            logger.debug(f"  Position {pos} ({name}): {actual} (expected {expected_id}) - {status}")
        
        for i, t in enumerate(tokens):
            if t == cond_end_id:
                logger.debug(f"  Position {i} (COND_END): {t} - OK")
                break
        
        for i, t in enumerate(tokens):
            if t == sep_id:
                logger.debug(f"  Position {i} (SEP): {t} - OK")
                break
                
        for i, t in enumerate(tokens):
            if t == mol_id:
                logger.debug(f"  Position {i} (MOL): {t} - OK")
                break
        
        if all_ok:
            logger.debug(f"  Template format VERIFIED")
        else:
            logger.warning(f"  Template format has MISMATCHES")
