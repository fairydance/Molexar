"""Shared Molexar template helpers."""

from typing import Dict, List, Tuple

from molexar.modeling import MolexarConfig


CONDITION_TOKEN_ATTRS: List[Tuple[str, str]] = [
    ("mol_hac", "MOL_HAC_TOKEN"),
    ("mol_hbdc", "MOL_HBDC_TOKEN"),
    ("mol_hbac", "MOL_HBAC_TOKEN"),
    ("mol_rotbc", "MOL_ROTBC_TOKEN"),
    ("mol_wt", "MOL_WT_TOKEN"),
    ("mol_logp", "MOL_LOGP_TOKEN"),
    ("mol_tpsa", "MOL_TPSA_TOKEN"),
    ("mol_qed", "MOL_QED_TOKEN"),
    ("mol_sas", "MOL_SAS_TOKEN"),
    ("mol_pharma_fp", "MOL_PHARMA_FP_TOKEN"),
    ("prot_seq_esm_emb", "PROT_SEQ_ESM_EMB_TOKEN"),
    ("prot_poc_gvp_emb", "PROT_POC_GVP_EMB_TOKEN"),
    ("unused0", "UNUSED0_TOKEN"),
    ("unused1", "UNUSED1_TOKEN"),
    ("unused2", "UNUSED2_TOKEN"),
    ("unused3", "UNUSED3_TOKEN"),
    ("unused4", "UNUSED4_TOKEN"),
    ("unused5", "UNUSED5_TOKEN"),
    ("unused6", "UNUSED6_TOKEN"),
    ("unused7", "UNUSED7_TOKEN"),
    ("unused8", "UNUSED8_TOKEN"),
    ("unused9", "UNUSED9_TOKEN"),
]


def build_condition_template(config: MolexarConfig) -> Tuple[str, Dict[str, int]]:
    """Build the unified condition block and absolute VALUE token positions."""
    parts = [config.COND_TOKEN]
    value_positions = {}

    for index, (key, token_attr) in enumerate(CONDITION_TOKEN_ATTRS):
        parts.append(f"{getattr(config, token_attr)}{config.VALUE_TOKEN}")
        value_positions[key] = 3 + index * 2

    parts.append(config.COND_END_TOKEN)
    return "".join(parts), value_positions


def build_generation_prompt(
    config: MolexarConfig,
    condition_block: str,
    start_string: str = "",
) -> str:
    """Build a generation prompt that starts inside the molecule block."""
    start = start_string.strip() if start_string else ""
    return (
        f"{config.BOS_TOKEN}"
        f"{condition_block}"
        f"{config.SEP_TOKEN}"
        f"{config.MOL_TOKEN}"
        f"{start}"
    )


def build_training_text(config: MolexarConfig, condition_block: str, fragment_selfies: str) -> str:
    """Build the full training template for one molecule."""
    return (
        f"{config.BOS_TOKEN}"
        f"{condition_block}"
        f"{config.SEP_TOKEN}"
        f"{config.MOL_TOKEN}"
        f"{fragment_selfies}"
        f"{config.MOL_END_TOKEN}"
        f"{config.EOS_TOKEN}"
    )


def strip_to_molecule(decoded_text: str, mol_token: str) -> str:
    """Keep only generated text after the Molexar molecule token."""
    if mol_token in decoded_text:
        decoded_text = decoded_text.split(mol_token, 1)[-1]
    return decoded_text.strip()
