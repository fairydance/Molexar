import argparse
from functools import lru_cache
from typing import Optional

from rdkit import Chem

try:
    from fragment_selfies import FragmentSelfiesCodec
    from fragment_selfies.exceptions import FragmentSelfiesError
    from fragment_selfies.tokens import split_tokens
except ImportError as exc:  # pragma: no cover - exercised by CLI users without dependency
    raise ImportError(
        "Fragment-SELFIES is required. Install the local Fragment-SELFIES package "
        "into this environment before using Molexar conversion utilities."
    ) from exc


class FragmentSelfiesConversionError(Exception):
    """Raised when Molexar cannot encode or decode Fragment-SELFIES."""


@lru_cache(maxsize=1)
def get_fragment_selfies_codec() -> FragmentSelfiesCodec:
    """Load a compact Fragment-SELFIES codec with process-local caching."""
    return FragmentSelfiesCodec()


def _mol_from_input(inp) -> Chem.Mol:
    mol = Chem.MolFromSmiles(inp) if isinstance(inp, str) else Chem.Mol(inp)
    if mol is None:
        raise ValueError(f"Invalid molecule: {inp}")
    return mol


def _has_dummy_attachment(mol: Chem.Mol) -> bool:
    return any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms())


def smiles_to_fragment_selfies(
    inp,
    canonical: bool = False,
    randomized: bool = False,
    seed: Optional[int] = None,
    fallback_selfies: bool = False,
    implicit_probability: Optional[float] = None,
    max_implicit_cuts: Optional[int] = None,
    fragment_style: Optional[str] = None,
    slicer: Optional[str] = None,
    require_hs: bool = False,
    ignore_stereo: bool = True,
) -> str:
    """Encode a SMILES string or RDKit molecule as Fragment-SELFIES.

    Fragment-SELFIES always uses compact BRICS fragments encoded as standard
    SELFIES token runs.
    """
    if canonical and randomized:
        raise ValueError("canonical=True and randomized=True are mutually exclusive")
    if not ignore_stereo:
        raise ValueError("Compact Fragment-SELFIES does not support ignore_stereo=False")
    if slicer not in (None, "brics"):
        raise ValueError("Fragment-SELFIES only supports the BRICS slicer")
    if require_hs:
        raise ValueError("Fragment-SELFIES does not support require_hs=True")

    codec = get_fragment_selfies_codec()

    try:
        mol = _mol_from_input(inp)
        if _has_dummy_attachment(mol):
            return codec.encode_fragment(
                mol,
                canonical=canonical,
                randomized=randomized,
                seed=seed,
            )
        encode_kwargs = {
            "fallback_selfies": fallback_selfies,
            "canonical": canonical,
            "randomized": randomized,
            "seed": seed,
        }
        if implicit_probability is not None:
            encode_kwargs["implicit_probability"] = implicit_probability
        if max_implicit_cuts is not None:
            encode_kwargs["max_implicit_cuts"] = max_implicit_cuts
        if fragment_style is not None:
            encode_kwargs["fragment_style"] = fragment_style
        return codec.encode(
            mol,
            **encode_kwargs,
        )
    except (FragmentSelfiesError, ValueError) as exc:
        raise FragmentSelfiesConversionError(f"Failed to encode molecule as Fragment-SELFIES: {inp}") from exc


def smiles_fragment_to_fragment_selfies(
    inp,
    canonical: bool = False,
    randomized: bool = False,
    seed: Optional[int] = None,
) -> str:
    """Encode a connected SMILES fragment, preserving dummy attachment atoms."""
    if canonical and randomized:
        raise ValueError("canonical=True and randomized=True are mutually exclusive")

    codec = get_fragment_selfies_codec()
    try:
        return codec.encode_fragment(
            _mol_from_input(inp),
            canonical=canonical,
            randomized=randomized,
            seed=seed,
        )
    except (FragmentSelfiesError, ValueError) as exc:
        raise FragmentSelfiesConversionError(
            f"Failed to encode SMILES fragment as Fragment-SELFIES: {inp}"
        ) from exc


def fragment_selfies_to_mol(
    fragment_selfies: str,
    strict: bool = False,
) -> Chem.Mol:
    """Decode Fragment-SELFIES to an RDKit molecule."""
    codec = get_fragment_selfies_codec()

    try:
        return codec.decode(fragment_selfies, strict=strict)
    except (FragmentSelfiesError, ValueError) as exc:
        raise FragmentSelfiesConversionError(
            f"Failed to decode Fragment-SELFIES: {fragment_selfies}"
        ) from exc


def fragment_selfies_to_smiles(
    fragment_selfies: str,
    as_mol: bool = False,
    canonical: bool = False,
    randomized: bool = False,
    strict: bool = False,
    ignore_errors: bool = False,
):
    """Decode Fragment-SELFIES to canonical non-isomeric SMILES or an RDKit molecule."""
    try:
        mol = fragment_selfies_to_mol(
            fragment_selfies,
            strict=strict,
        )
    except FragmentSelfiesConversionError:
        if ignore_errors:
            return None
        raise

    if as_mol:
        return mol
    return Chem.MolToSmiles(mol, canonical=canonical, doRandom=randomized, isomericSmiles=False)


def fragment_selfies_fragment_to_mol(fragment_selfies: str) -> Chem.Mol:
    """Decode a Fragment-SELFIES fragment while preserving dummy attachment atoms."""
    codec = get_fragment_selfies_codec()

    try:
        return codec.decode_fragment(fragment_selfies)
    except (FragmentSelfiesError, ValueError) as exc:
        raise FragmentSelfiesConversionError(
            f"Failed to decode Fragment-SELFIES fragment: {fragment_selfies}"
        ) from exc


def fragment_selfies_fragment_to_smiles(
    fragment_selfies: str,
    canonical: bool = False,
    randomized: bool = False,
) -> str:
    """Decode a Fragment-SELFIES fragment to SMILES with dummy atoms preserved."""
    mol = fragment_selfies_fragment_to_mol(fragment_selfies)
    return Chem.MolToSmiles(mol, canonical=canonical, doRandom=randomized, isomericSmiles=False)


def tokenize_fragment_selfies(fragment_selfies: str) -> list[str]:
    """Split a Fragment-SELFIES string into bracketed tokens."""
    return split_tokens(fragment_selfies)


def validate_fragment_selfies(
    fragment_selfies: str,
    strict: bool = False,
) -> bool:
    """Return True when a Fragment-SELFIES string decodes successfully."""
    try:
        fragment_selfies_to_mol(
            fragment_selfies,
            strict=strict,
        )
        return True
    except FragmentSelfiesConversionError:
        return False


def show_example() -> None:
    examples = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "c1ccccc1C(=O)O",
        "CC(=O)O",
    ]

    for smiles in examples:
        encoded = smiles_to_fragment_selfies(
            smiles,
            randomized=False,
        )
        decoded = fragment_selfies_to_smiles(encoded)
        print(f"Original SMILES:       {smiles}")
        print(f"Fragment-SELFIES:      {encoded}")
        print(f"Recovered SMILES:      {decoded}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="SMILES/Fragment-SELFIES converter")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--to-fragment-selfies", action="store_true", help="Encode SMILES to Fragment-SELFIES")
    mode.add_argument("--from-fragment-selfies", action="store_true", help="Decode Fragment-SELFIES to SMILES")

    parser.add_argument("-i", "--input")
    parser.add_argument("-e", "--example", action="store_true")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--canonical", action="store_true", help="Use canonical encoding or SMILES output")
    output_mode.add_argument("--randomized", action="store_true", help="Use randomized encoding or SMILES output")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--implicit-probability", type=float, default=None)
    parser.add_argument("--max-implicit-cuts", type=int, default=None)
    parser.add_argument("--fragment-style", choices=["auto", "explicit", "implicit"], default=None)
    parser.add_argument("--fallback-selfies", action="store_true")
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument("--as-mol", action="store_true")
    parser.add_argument("--ignore-errors", action="store_true")
    parser.add_argument(
        "--fragment",
        action="store_true",
        help="Preserve dummy atoms as fragment attachment points",
    )

    args = parser.parse_args()

    try:
        if args.example:
            show_example()
            return

        if not args.input:
            parser.error("Input required (-i)")

        if args.from_fragment_selfies:
            if args.fragment:
                if args.as_mol:
                    result = fragment_selfies_fragment_to_mol(args.input)
                else:
                    result = fragment_selfies_fragment_to_smiles(
                        args.input,
                        canonical=args.canonical,
                        randomized=args.randomized,
                    )
            else:
                result = fragment_selfies_to_smiles(
                    args.input,
                    as_mol=args.as_mol,
                    canonical=args.canonical,
                    randomized=args.randomized,
                    strict=args.strict,
                    ignore_errors=args.ignore_errors,
                )
            if args.as_mol and result is not None:
                print(Chem.MolToSmiles(result, canonical=args.canonical, isomericSmiles=False))
            elif result is not None:
                print(result)
        else:
            if args.fragment:
                result = smiles_fragment_to_fragment_selfies(
                    args.input,
                    canonical=args.canonical,
                    randomized=args.randomized,
                    seed=args.seed,
                )
            else:
                result = smiles_to_fragment_selfies(
                    args.input,
                    canonical=args.canonical,
                    randomized=args.randomized,
                    seed=args.seed,
                    fallback_selfies=args.fallback_selfies,
                    implicit_probability=args.implicit_probability,
                    max_implicit_cuts=args.max_implicit_cuts,
                    fragment_style=args.fragment_style,
                )
            print(result)
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
