"""Unified Molexar inference utilities."""

import importlib.util
import json
import os
import pickle
import random
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from molexar.datasets.pocket import PocketProcessor
from molexar.modeling import MolexarConfig, MolexarForCausalLM
from molexar.templates import build_condition_template, build_generation_prompt, strip_to_molecule
from molexar.tokenizer import MolexarTokenizerFast

try:
    from molexar.data.converter import fragment_selfies_to_smiles
except ImportError:  # pragma: no cover - conversion is optional
    fragment_selfies_to_smiles = None


PROPERTY_KEYS = [
    "mol_hac",
    "mol_hbdc",
    "mol_hbac",
    "mol_rotbc",
    "mol_wt",
    "mol_logp",
    "mol_tpsa",
    "mol_qed",
    "mol_sas",
]
DISCRETE_PROPERTY_KEYS = {"mol_hac", "mol_hbdc", "mol_hbac", "mol_rotbc"}
DEFAULT_MAX_NEW_TOKENS = 204
PHARMA_FP_DIM = 1032
ESM_EMBEDDING_DIM = 1152
DEFAULT_ESM_CONDA_ENV = "molexar_aux"


def count_tokenized_start_string(tokenizer, start_string: Optional[str]) -> int:
    start = start_string.strip() if start_string else ""
    if not start:
        return 0
    input_ids = tokenizer(start, add_special_tokens=False)["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def resolve_max_new_tokens(max_new_tokens: Optional[int], tokenizer, start_string: Optional[str]) -> int:
    if max_new_tokens is not None:
        return max_new_tokens
    start_token_count = count_tokenized_start_string(tokenizer, start_string)
    resolved = DEFAULT_MAX_NEW_TOKENS - start_token_count
    if resolved < 1:
        raise ValueError(
            f"--start_string uses {start_token_count} tokens, leaving no room under "
            f"the default {DEFAULT_MAX_NEW_TOKENS}-token generation budget"
        )
    return resolved


class MolexarInference:
    """One inference engine for base and conditional Molexar generation."""

    def __init__(self, model_path: str, device: str = "auto", tokenizer_path: Optional[str] = None):
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.tokenizer = self._load_tokenizer(model_path, tokenizer_path)
        self.config = MolexarConfig.from_pretrained(model_path)
        self.config.use_cache = True
        self.model = MolexarForCausalLM.from_pretrained(model_path, config=self.config)
        self.model.to(self.device)
        self.model.eval()
        self.cond_block, self.value_positions = build_condition_template(self.config)
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def generate(
        self,
        conditions: Optional[Dict[str, Any]] = None,
        start_string: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        num_samples: int = 1,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        batch_size: int = 100,
        batch_delay: float = 0.0,
    ) -> List[str]:
        max_new_tokens = resolve_max_new_tokens(max_new_tokens, self.tokenizer, start_string)
        prompt = build_generation_prompt(self.config, self.cond_block, start_string or "")
        input_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        conditions = normalize_conditions(conditions or {})

        all_results = []
        num_batches = (num_samples + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            batch_num_samples = min(batch_size, num_samples - batch_idx * batch_size)
            condition_values, condition_indices = self._build_condition_batch(conditions, batch_num_samples)
            gen_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "num_return_sequences": batch_num_samples,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": repetition_penalty,
            }
            if do_sample:
                gen_kwargs.update({"temperature": temperature, "top_p": top_p, "top_k": top_k})
            if condition_values:
                gen_kwargs["condition_values"] = condition_values
                gen_kwargs["condition_indices"] = condition_indices
            with torch.no_grad():
                outputs = self.model.generate(**gen_kwargs)
            for output in outputs:
                decoded = self.tokenizer.decode(output, skip_special_tokens=True)
                all_results.append(strip_to_molecule(decoded, self.config.MOL_TOKEN))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if batch_delay > 0 and batch_idx < num_batches - 1:
                time.sleep(batch_delay)
        return all_results

    def convert_to_smiles(
        self,
        fragment_selfies_strings: List[str],
        canonical: bool = False,
        randomized: bool = False,
        strict: bool = False,
    ):
        if fragment_selfies_to_smiles is None:
            raise ImportError("fragment_selfies is required to convert generated strings to SMILES")
        results = []
        for fragment_selfies in fragment_selfies_strings:
            smiles = None
            try:
                smiles = fragment_selfies_to_smiles(
                    fragment_selfies,
                    canonical=canonical,
                    randomized=randomized,
                    strict=strict,
                    ignore_errors=False,
                )
            except Exception as exc:
                logger.warning(f"Conversion failed for '{fragment_selfies}': {exc}")
            results.append((fragment_selfies, smiles))
        return results

    def _build_condition_batch(self, conditions: Dict[str, Any], batch_size: int):
        condition_values = {}
        condition_indices = {}
        for key, value in conditions.items():
            if key not in self.value_positions:
                logger.warning(f"Ignoring unknown condition key: {key}")
                continue
            if isinstance(value, dict):
                condition_values[key] = move_pocket_graph(value, self.device)
            else:
                tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.float32)
                if tensor.dim() == 0:
                    tensor = tensor.unsqueeze(0)
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)
                expected_dim = self.config.condition_settings.get(key, {}).get("dim")
                if expected_dim is not None and tensor.shape[-1] != expected_dim:
                    raise ValueError(
                        f"Condition {key} expects dimension {expected_dim}, got {tensor.shape[-1]}"
                    )
                tensor = tensor.to(self.device)
                if tensor.shape[0] != batch_size:
                    tensor = tensor.expand(batch_size, -1).clone()
                condition_values[key] = tensor
            condition_indices[key] = [self.value_positions[key]] * batch_size
        return condition_values, condition_indices

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_tokenizer(self, model_path: str, tokenizer_path: Optional[str]):
        candidates = []
        if tokenizer_path:
            candidates.extend([tokenizer_path, os.path.join(tokenizer_path, "tokenizer.json")])
        candidates.extend([
            model_path,
            os.path.join(model_path, "tokenizer.json"),
            os.path.join(os.path.dirname(model_path), "tokenizer"),
        ])
        for candidate in candidates:
            if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "tokenizer.json")):
                return MolexarTokenizerFast(tokenizer_file=os.path.join(candidate, "tokenizer.json"))
            if os.path.isfile(candidate):
                return MolexarTokenizerFast(tokenizer_file=candidate)
        raise FileNotFoundError("tokenizer.json not found; pass --tokenizer_path")


def normalize_conditions(conditions: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {}
    for key, value in conditions.items():
        if key in DISCRETE_PROPERTY_KEYS and not isinstance(value, (list, tuple, dict, torch.Tensor, np.ndarray)):
            normalized[key] = [int(round(float(value)))]
        elif isinstance(value, np.ndarray):
            normalized[key] = torch.tensor(value, dtype=torch.float32)
        elif isinstance(value, (int, float)):
            normalized[key] = [float(value)]
        else:
            normalized[key] = value
    return normalized


def load_condition_file(path: str, condition_key: Optional[str] = None) -> Dict[str, Any]:
    """Load conditions from JSON, PKL, or NPY files."""
    if path.endswith(".json"):
        with open(path, "r") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data.get("conditions", data)
        if not condition_key:
            raise ValueError("--condition_key is required when a JSON condition file is not an object")
        return {condition_key: data}
    if path.endswith((".pkl", ".pickle")):
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        return data.get("conditions", data) if isinstance(data, dict) else {condition_key: data}
    if path.endswith(".npy"):
        if not condition_key:
            raise ValueError("--condition_key is required when loading a .npy condition file")
        value = np.load(path)
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if condition_key == "mol_pharma_fp" and value.shape[-1] != 1032:
            value = np.unpackbits(value.reshape(1, -1), axis=1, count=1032)[0].astype(np.float32)
        return {condition_key: value}
    raise ValueError(f"Unsupported condition file: {path}")


def load_conditions_from_args(args, config: Optional[MolexarConfig] = None) -> Dict[str, Any]:
    conditions = {}
    if args.condition_json:
        if os.path.isfile(args.condition_json):
            with open(args.condition_json, "r") as handle:
                data = json.load(handle)
        else:
            data = json.loads(args.condition_json)
        conditions.update(data.get("conditions", data))
    if args.condition_file:
        conditions.update(load_condition_file(args.condition_file, args.condition_key))
    for key in PROPERTY_KEYS:
        value = getattr(args, key, None)
        if value is not None:
            conditions[key] = int(value) if key in DISCRETE_PROPERTY_KEYS else float(value)
    if args.reference_smiles:
        if args.condition_key == "mol_pharma_fp":
            conditions["mol_pharma_fp"] = compute_pharma_fp_from_smiles(args.reference_smiles)
        else:
            conditions.update(
                filter_property_conditions(
                    compute_properties_from_smiles(args.reference_smiles),
                    args.property_keys,
                    args.num_random_properties,
                )
            )
    if args.pharma_fp_file:
        key = args.condition_key or "mol_pharma_fp"
        conditions.update(load_condition_file(args.pharma_fp_file, key))
    if args.esm_file:
        if getattr(args, "protein_sequence", None) or getattr(args, "protein_sequence_file", None):
            raise ValueError("Use only one of --esm_file, --protein_sequence, or --protein_sequence_file")
        conditions.update(load_condition_file(args.esm_file, args.condition_key or "prot_seq_esm_emb"))
    if getattr(args, "protein_sequence", None) or getattr(args, "protein_sequence_file", None):
        conditions["prot_seq_esm_emb"] = compute_esm_embedding_condition(
            sequence=getattr(args, "protein_sequence", None),
            input_path=getattr(args, "protein_sequence_file", None),
            conda_env=getattr(args, "esm_conda_env", DEFAULT_ESM_CONDA_ENV),
            device=getattr(args, "esm_device", "cuda"),
            script_path=getattr(args, "esm_embedding_script", None),
        )
    if args.pocket_pdb:
        node_scalar_dim = config.gvp_node_in_dim[0] if config is not None else None
        processor = PocketProcessor(
            pocket_radius=args.pocket_radius,
            max_atoms=args.max_atoms,
            include_hydrogens=False,
            node_scalar_dim=node_scalar_dim,
        )
        conditions["prot_poc_gvp_emb"] = processor.process_pdb(args.pocket_pdb)
    return conditions


def compute_esm_embedding_condition(
    sequence: Optional[str] = None,
    input_path: Optional[str] = None,
    conda_env: str = DEFAULT_ESM_CONDA_ENV,
    device: str = "cuda",
    script_path: Optional[str] = None,
) -> np.ndarray:
    if bool(sequence) == bool(input_path):
        raise ValueError("Provide exactly one of --protein_sequence or --protein_sequence_file")

    embedding_script = Path(script_path) if script_path else _default_esm_embedding_script()
    if not embedding_script.exists():
        raise FileNotFoundError(f"ESM embedding script not found: {embedding_script}")

    with tempfile.TemporaryDirectory(prefix="molexar_esm_") as temp_dir:
        output_path = Path(temp_dir) / "esm_embedding.pkl"
        command = [
            "conda",
            "run",
            "-n",
            conda_env,
            "python",
            str(embedding_script),
            "--output",
            str(output_path),
            "--format",
            "pkl",
            "--storage",
            "numpy",
            "--device",
            device,
        ]
        if sequence:
            command.extend(["--sequence", sequence])
            source = f"inline sequence length {len(''.join(sequence.split()))}"
        else:
            command.extend(["--input", str(input_path)])
            source = str(input_path)

        logger.info(f"Computing ESM embedding from {source} using conda env '{conda_env}'")
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("Could not run conda; ensure conda is available on PATH") from exc

        if result.returncode != 0:
            message = (result.stderr or result.stdout or "ESM embedding command failed").strip()
            raise RuntimeError(f"ESM embedding command failed with exit code {result.returncode}: {message[-2000:]}")
        if result.stderr:
            logger.debug(result.stderr.strip())
        if result.stdout:
            logger.debug(result.stdout.strip())

        payload = load_condition_file(str(output_path), "prot_seq_esm_emb")
        embedding = np.asarray(payload["prot_seq_esm_emb"], dtype=np.float32).reshape(-1)
        if embedding.shape[0] != ESM_EMBEDDING_DIM:
            raise ValueError(f"Expected ESM embedding dimension {ESM_EMBEDDING_DIM}, got {embedding.shape[0]}")
        return embedding


def _default_esm_embedding_script() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "embeddings" / "compute_esm_embedding.py"


def compute_pharma_fp_from_smiles(smiles: str) -> np.ndarray:
    try:
        from rdkit import Chem
        from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
    except ImportError as exc:
        raise ImportError("RDKit is required to compute pharmacophore fingerprints from SMILES") from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    factory = Gobbi_Pharm2D.factory
    factory.SetBins([(0, 3), (3, 8)])
    factory.minPointCount = 2
    factory.maxPointCount = 3
    factory.Init()

    fp = Generate.Gen2DFingerprint(mol, factory)
    array = np.zeros((fp.GetNumBits(),), dtype=np.float32)
    for bit in fp.GetOnBits():
        array[bit] = 1.0
    if array.shape[0] != PHARMA_FP_DIM:
        raise ValueError(f"Unexpected pharmacophore fingerprint dimension: {array.shape[0]} != {PHARMA_FP_DIM}")
    return array


@lru_cache(maxsize=1)
def _load_sa_scorer():
    from rdkit import RDConfig

    scorer_path = Path(RDConfig.RDContribDir) / "SA_Score" / "sascorer.py"
    spec = importlib.util.spec_from_file_location("rdkit_sa_scorer", scorer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load RDKit SA scorer from {scorer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_properties_from_smiles(smiles: str) -> Dict[str, Any]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED

        sascorer = _load_sa_scorer()
    except ImportError as exc:
        raise ImportError("RDKit is required for --reference_smiles property conditioning") from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return {
        "mol_hac": int(mol.GetNumHeavyAtoms()),
        "mol_hbdc": int(Descriptors.NumHDonors(mol)),
        "mol_hbac": int(Descriptors.NumHAcceptors(mol)),
        "mol_rotbc": int(Descriptors.NumRotatableBonds(mol)),
        "mol_wt": float(Descriptors.MolWt(mol)),
        "mol_logp": float(Descriptors.MolLogP(mol)),
        "mol_tpsa": float(Descriptors.TPSA(mol)),
        "mol_qed": float(QED.qed(mol)),
        "mol_sas": float(sascorer.calculateScore(mol)),
    }


def filter_property_conditions(
    properties: Dict[str, Any],
    property_keys: Optional[str],
    num_random_properties: Optional[int] = None,
) -> Dict[str, Any]:
    if num_random_properties is not None:
        if num_random_properties < 0 or num_random_properties > len(PROPERTY_KEYS):
            raise ValueError(f"--num_random_properties must be between 0 and {len(PROPERTY_KEYS)}")
        selected = set(random.sample(PROPERTY_KEYS, num_random_properties))
        return {key: value for key, value in properties.items() if key in selected}
    if not property_keys:
        return properties
    selected = {key.strip() for key in property_keys.split(",") if key.strip()}
    unknown = selected.difference(PROPERTY_KEYS)
    if unknown:
        raise ValueError(f"Unknown property keys: {sorted(unknown)}")
    return {key: value for key, value in properties.items() if key in selected}


def move_pocket_graph(value: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
    result = {}
    for key, item in value.items():
        if key in {"n_atoms", "center"}:
            continue
        tensor = item if isinstance(item, torch.Tensor) else torch.tensor(item)
        result[key] = tensor.to(device)
    return result
