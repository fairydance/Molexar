"""Helpers for extracting Fragment-SELFIES start strings."""

from typing import List, Optional

from fragment_selfies.tokens import (
    DOT_TOKEN,
    FRAGMENT_START,
    POP_TOKEN,
    SELFIES_END,
    SELFIES_START,
    parse_attachment_token,
    parse_fragment_token,
    split_tokens,
)

from molexar.data.converter import fragment_selfies_to_smiles, smiles_to_fragment_selfies


def is_control_token(token: str) -> bool:
    """Return True for Fragment-SELFIES control tokens outside fragment payloads."""
    return (
        token in {POP_TOKEN, DOT_TOKEN, SELFIES_START, SELFIES_END}
        or parse_attachment_token(token) is not None
        or parse_fragment_token(token) is not None
    )


def detect_input_format(value: str, requested: str) -> str:
    """Resolve auto input format detection for SMILES or Fragment-SELFIES."""
    if requested != "auto":
        return requested
    if value.startswith(("[Frag]", "[Frag@", SELFIES_START)):
        return "fragment_selfies"
    return "smiles"


def encode_molecule(
    value: str,
    input_format: str,
    canonical: bool,
    randomized: bool,
    seed: Optional[int],
    implicit_probability: Optional[float] = None,
    max_implicit_cuts: Optional[int] = None,
    fragment_style: Optional[str] = None,
) -> str:
    """Return compact Fragment-SELFIES for a SMILES or Fragment-SELFIES input."""
    if input_format == "fragment_selfies":
        return value
    return smiles_to_fragment_selfies(
        value,
        canonical=canonical,
        randomized=randomized,
        seed=seed,
        fallback_selfies=False,
        implicit_probability=implicit_probability,
        max_implicit_cuts=max_implicit_cuts,
        fragment_style=fragment_style,
    )


def extract_start_strings(fragment_selfies: str) -> List[str]:
    """Extract individual fragments normalized as root `[Frag]...` start strings."""
    tokens = split_tokens(fragment_selfies)
    start_strings = []
    idx = 0
    while idx < len(tokens):
        fragment_token = parse_fragment_token(tokens[idx])
        if fragment_token is None:
            idx += 1
            continue

        idx += 1
        fragment_tokens = []
        while idx < len(tokens) and not is_control_token(tokens[idx]):
            fragment_tokens.append(tokens[idx])
            idx += 1
        if fragment_tokens:
            start_strings.append("".join([FRAGMENT_START, *fragment_tokens]))
    return start_strings


def start_string_to_smiles(start_string: str, strict: bool) -> str:
    """Decode a start string to canonical SMILES for validation/reporting."""
    return fragment_selfies_to_smiles(
        start_string,
        canonical=True,
        randomized=False,
        strict=strict,
        ignore_errors=False,
    )
