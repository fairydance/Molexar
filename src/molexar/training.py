"""Shared training utilities for Molexar pretraining and SFT."""

import csv
import hashlib
import json
import math
import os
import pickle
import random
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import Trainer, TrainerCallback, TrainingArguments

from molexar.datasets import PretrainingDataset
from molexar.datasets.pocket import (
    PocketProcessor,
    resolve_plinder_pocket_path,
    resolve_sair_pocket_path,
)
from molexar.modeling import MolexarConfig, MolexarForCausalLM
from molexar.templates import build_condition_template, build_training_text
from molexar.tokenizer import MolexarTokenizerFast, train_and_save_tokenizer

try:
    from molexar.data.converter import fragment_selfies_to_smiles, smiles_to_fragment_selfies
except ImportError:  # pragma: no cover - optional until SMILES conversion is requested
    fragment_selfies_to_smiles = None
    smiles_to_fragment_selfies = None


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
PROPERTY_COLUMN_MAP = {
    "mol_hac": "HAC",
    "mol_hbdc": "HBDC",
    "mol_hbac": "HBAC",
    "mol_rotbc": "ROTBC",
    "mol_wt": "WT",
    "mol_logp": "LOGP",
    "mol_tpsa": "TPSA",
    "mol_qed": "QED",
    "mol_sas": "SAS",
}
MISSING_CONDITION_INDEX = 10 ** 9
MOLECULE_CONDITION_KEYS = PROPERTY_KEYS + ["mol_pharma_fp"]
DEFAULT_SINGLE_CONDITION_WEIGHT = 0.6
DEFAULT_DUAL_CONDITION_WEIGHT = 0.3
DEFAULT_TRIPLE_CONDITION_WEIGHT = 0.1
DEFAULT_MOLECULE_PHARMA_FP_CONDITION_PROBABILITY = 0.5
DEFAULT_TARGET_DUAL_CONTEXT_PROBABILITY = 0.2
DEFAULT_TARGET_LIGAND_CONDITION_PROBABILITY = 0.2


def _seed_dataloader_worker(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    num_workers = worker_info.num_workers if worker_info is not None else 1
    rank = 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    seed = (num_workers * rank + torch.initial_seed()) % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sample_condition_count(available_count: int, args, max_count: int = 3) -> int:
    if available_count <= 0:
        return 0
    explicit_max = getattr(args, "max_condition_count", None)
    if explicit_max is not None:
        max_count = min(max_count, explicit_max)
    max_count = min(max_count, available_count)
    if max_count <= 0:
        return 0

    weights_by_count = {
        1: getattr(args, "single_condition_weight", DEFAULT_SINGLE_CONDITION_WEIGHT),
        2: getattr(args, "dual_condition_weight", DEFAULT_DUAL_CONDITION_WEIGHT),
        3: getattr(args, "triple_condition_weight", DEFAULT_TRIPLE_CONDITION_WEIGHT),
    }
    counts = [count for count in (1, 2, 3) if count <= max_count]
    weights = [max(0.0, float(weights_by_count[count])) for count in counts]
    if not any(weights):
        return counts[0]
    return random.choices(counts, weights=weights, k=1)[0]


def _sample_condition_keys(keys: Sequence[str], args, max_count: int = 3) -> List[str]:
    keys = list(keys)
    count = _sample_condition_count(len(keys), args, max_count=max_count)
    return random.sample(keys, count) if count else []


def _sample_molecule_context_keys(keys: Sequence[str], args, max_count: int = 3) -> List[str]:
    keys = list(keys)
    count = _sample_condition_count(len(keys), args, max_count=max_count)
    if count <= 0:
        return []

    pharma_probability = float(getattr(
        args,
        "molecule_pharma_fp_condition_probability",
        DEFAULT_MOLECULE_PHARMA_FP_CONDITION_PROBABILITY,
    ))
    if "mol_pharma_fp" not in keys or random.random() >= pharma_probability:
        return random.sample(keys, count)

    selected = ["mol_pharma_fp"]
    remaining = [key for key in keys if key != "mol_pharma_fp"]
    selected.extend(random.sample(remaining, min(count - 1, len(remaining))))
    return selected


class OffsetLineReader:
    """Random-access text line reader backed by byte offsets."""

    def __init__(self, path: str, skip_lines: int = 0, max_lines: Optional[int] = None):
        self.path = path
        self.offsets = _load_line_offsets(path, skip_lines=skip_lines, max_lines=max_lines)
        self._handle = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def line(self, index: int) -> str:
        if self._handle is None:
            self._handle = open(self.path, "rb")
        self._handle.seek(int(self.offsets[index]))
        return self._handle.readline().decode("utf-8", errors="replace").strip()


def _load_line_offsets(path: str, skip_lines: int = 0, max_lines: Optional[int] = None) -> np.ndarray:
    cache_path = f"{path}.skip{skip_lines}.offsets.npy" if max_lines is None else None
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading line-offset cache: {cache_path}")
        try:
            return np.load(cache_path, mmap_mode="r")
        except (EOFError, OSError, ValueError) as exc:
            logger.warning(f"Ignoring invalid line-offset cache {cache_path}: {exc}")

    logger.info(f"Building line offsets for {path}")
    offsets = []
    with open(path, "rb") as handle:
        for _ in range(skip_lines):
            handle.readline()
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
                if max_lines is not None and len(offsets) >= max_lines:
                    break
    offsets_array = np.asarray(offsets, dtype=np.uint64)
    if cache_path:
        temp_path = None
        try:
            cache_dir = os.path.dirname(cache_path) or "."
            cache_name = os.path.basename(cache_path)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=cache_dir,
                prefix=f".{cache_name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = handle.name
                np.save(handle, offsets_array)
            os.replace(temp_path, cache_path)
            logger.info(f"Saved line-offset cache: {cache_path}")
            return np.load(cache_path, mmap_mode="r")
        except OSError as exc:
            logger.warning(f"Could not save line-offset cache {cache_path}: {exc}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
    return offsets_array


class SmilesMoleculeConditionLookup:
    """Load target ligand molecule data by exact SMILES."""

    def __init__(
        self,
        smiles_path: str,
        properties_path: str,
        fp_path: str,
        config: MolexarConfig,
        label: str,
        fragment_selfies_path: Optional[str] = None,
        fragment_selfies_folds: int = 10,
    ):
        for name, path in (("SMILES", smiles_path), ("properties", properties_path), ("pharma_fp", fp_path)):
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"Missing {label} {name} file: {path}")
        if fragment_selfies_path and not os.path.exists(fragment_selfies_path):
            raise FileNotFoundError(f"Missing {label} Fragment-SELFIES file: {fragment_selfies_path}")

        self.smiles_path = smiles_path
        self.properties_path = properties_path
        self.fp_path = fp_path
        self.fragment_selfies_path = fragment_selfies_path
        self.fragment_selfies = OffsetLineReader(fragment_selfies_path) if fragment_selfies_path else None
        self.fragment_selfies_folds = 0
        self.pharma_fps = np.load(fp_path, mmap_mode="r")
        self.pharma_fp_dim = config.condition_settings.get("mol_pharma_fp", {}).get("dim", 1032)
        self.entries: Dict[str, Tuple[int, Dict[str, Any]]] = {}
        self.condition_row_count = 0

        with open(smiles_path, "r", encoding="utf-8") as smiles_handle, open(
            properties_path,
            "r",
            newline="",
            encoding="utf-8",
        ) as properties_handle:
            reader = csv.DictReader(properties_handle)
            for row_index, (smiles_line, row) in enumerate(zip(smiles_handle, reader)):
                self.condition_row_count = row_index + 1
                smiles = smiles_line.strip().split(maxsplit=1)[0]
                if not smiles or not _is_valid_condition_row(row):
                    continue
                if row_index >= self.pharma_fps.shape[0]:
                    logger.warning(f"Skipping {smiles}: no aligned pharmacophore row at {row_index}")
                    continue
                try:
                    self.entries[smiles] = (row_index, _property_conditions_from_row(row))
                except Exception as exc:
                    logger.warning(f"Skipping {smiles}: invalid precomputed properties: {exc}")

        if not self.entries:
            raise ValueError(f"No valid {label} SMILES condition rows loaded from {properties_path}")
        self._configure_fragment_selfies(label, fragment_selfies_folds)
        logger.info(
            f"Loaded {label} SMILES condition lookup: "
            f"entries={len(self.entries)}, smiles={smiles_path}, properties={properties_path}, "
            f"pharma_fp={fp_path}, fragment_selfies={fragment_selfies_path}"
        )

    def _configure_fragment_selfies(self, label: str, requested_folds: int) -> None:
        if self.fragment_selfies is None:
            return
        if self.condition_row_count <= 0:
            raise ValueError(f"No {label} ligand rows available for Fragment-SELFIES lookup")
        fragment_rows = len(self.fragment_selfies)
        if fragment_rows % self.condition_row_count != 0:
            raise ValueError(
                f"{label} Fragment-SELFIES rows must be an integer multiple of SMILES rows: "
                f"fragments={fragment_rows}, smiles={self.condition_row_count}"
            )
        available_folds = fragment_rows // self.condition_row_count
        if available_folds <= 0:
            raise ValueError(f"No {label} Fragment-SELFIES folds loaded from {self.fragment_selfies_path}")
        requested_folds = max(1, int(requested_folds))
        self.fragment_selfies_folds = min(requested_folds, available_folds)
        if self.fragment_selfies_folds != requested_folds:
            logger.warning(
                f"Using {self.fragment_selfies_folds} {label} Fragment-SELFIES folds; "
                f"requested {requested_folds}, available {available_folds}"
            )

    def get(self, smiles: str) -> Dict[str, Any]:
        entry = self.entries.get(smiles)
        if entry is None:
            return {}
        row_index, props = entry
        conditions = dict(props)
        conditions["mol_pharma_fp"] = _load_pharma_fp_row(
            self.pharma_fps,
            row_index,
            self.pharma_fp_dim,
        )
        return conditions

    def sample_fragment_selfies(self, smiles: str) -> str:
        if self.fragment_selfies is None:
            raise ValueError(f"No Fragment-SELFIES lookup configured for {self.smiles_path}")
        entry = self.entries.get(smiles)
        if entry is None:
            raise KeyError(f"SMILES not found in ligand Fragment-SELFIES lookup: {smiles}")
        row_index, _ = entry
        fold_index = random.randrange(self.fragment_selfies_folds)
        return self.fragment_selfies.line(fold_index * self.condition_row_count + row_index)


@dataclass
class OutputPaths:
    checkpoint_dir: str
    final_output_dir: str
    metrics_dir: str
    log_file: Optional[str]
    tokenizer_dir: str


def setup_logging(log_file: Optional[str], verbose: bool = False) -> None:
    """Configure loguru for command-line training."""
    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        logger.add(log_file, level=log_level, rotation="50 MB", encoding="utf-8", retention="10 days")


def resolve_output_paths(args, default_name: str) -> OutputPaths:
    """Resolve checkpoint/final/metrics/log paths from common CLI options."""
    tokenizer_output_dir = getattr(args, "tokenizer_output_dir", None)
    if args.output_dir:
        base_dir = args.output_dir
        checkpoint_dir = args.checkpoint_dir or os.path.join(base_dir, "checkpoints")
        final_output_dir = args.final_output_dir or os.path.join(base_dir, "final_model")
        metrics_dir = args.metrics_dir or os.path.join(base_dir, "metrics")
        log_file = args.log_file or os.path.join(base_dir, "training.log")
        tokenizer_dir = tokenizer_output_dir or os.path.join(base_dir, "tokenizer")
    else:
        base_dir = os.path.join("models", default_name)
        checkpoint_dir = args.checkpoint_dir or os.path.join(base_dir, "checkpoints")
        final_output_dir = args.final_output_dir or os.path.join(base_dir, "final_model")
        metrics_dir = args.metrics_dir or os.path.join(base_dir, "metrics")
        log_file = args.log_file
        tokenizer_dir = tokenizer_output_dir or os.path.join(base_dir, "tokenizer")

    for path in [checkpoint_dir, final_output_dir, metrics_dir, tokenizer_dir]:
        os.makedirs(path, exist_ok=True)
    if log_file and os.path.dirname(log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    return OutputPaths(checkpoint_dir, final_output_dir, metrics_dir, log_file, tokenizer_dir)


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return the newest Hugging Face checkpoint directory, if present."""
    if not os.path.isdir(checkpoint_dir):
        return None
    checkpoints = []
    for item in os.listdir(checkpoint_dir):
        match = re.match(r"checkpoint-(\d+)", item)
        if match:
            checkpoints.append((int(match.group(1)), os.path.join(checkpoint_dir, item)))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0], reverse=True)
    return checkpoints[0][1]


def repair_optimizer_checkpoint(checkpoint_dir: str) -> bool:
    """Add missing AdamW step tensors in optimizer checkpoints when needed."""
    optimizer_path = os.path.join(checkpoint_dir, "optimizer.bin")
    if not os.path.isfile(optimizer_path):
        return False

    optimizer_state = torch.load(optimizer_path, map_location="cpu", weights_only=False)
    state = optimizer_state.get("state") if isinstance(optimizer_state, dict) else None
    if not isinstance(state, dict):
        return False

    step_template = None
    for parameter_state in state.values():
        if isinstance(parameter_state, dict) and "step" in parameter_state:
            step = parameter_state["step"]
            if torch.is_tensor(step):
                step_template = step.detach().clone().cpu()
            else:
                step_template = torch.tensor(float(step), dtype=torch.float32)
            break

    if step_template is None:
        global_step = 0
        trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
        if os.path.isfile(trainer_state_path):
            with open(trainer_state_path) as handle:
                global_step = int(json.load(handle).get("global_step", 0))
        else:
            match = re.search(r"checkpoint-(\d+)$", os.path.basename(checkpoint_dir))
            if match:
                global_step = int(match.group(1))
        step_template = torch.tensor(float(global_step), dtype=torch.float32)

    repaired = 0
    for parameter_state in state.values():
        if isinstance(parameter_state, dict) and parameter_state and "step" not in parameter_state:
            parameter_state["step"] = step_template.clone()
            repaired += 1
    if repaired == 0:
        return False

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=os.path.dirname(optimizer_path),
            prefix=".optimizer.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = handle.name
        os.chmod(temp_path, os.stat(optimizer_path).st_mode)
        torch.save(optimizer_state, temp_path)
        os.replace(temp_path, optimizer_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    logger.warning(
        f"Repaired optimizer checkpoint {optimizer_path}: "
        f"added missing AdamW step to {repaired} state entries"
    )
    return True


def convert_smiles(smiles: str, args) -> str:
    """Convert one SMILES string to compact Fragment-SELFIES."""
    if smiles_to_fragment_selfies is None:
        raise ImportError("Fragment-SELFIES conversion is required when --data_format smiles is used")
    return smiles_to_fragment_selfies(
        smiles,
        canonical=args.canonical,
        randomized=args.randomized,
        slicer=args.slicer,
        require_hs=args.require_hs,
        ignore_stereo=args.ignore_stereo,
        fallback_selfies=getattr(args, "fallback_selfies", False),
        implicit_probability=getattr(args, "implicit_probability", None),
        max_implicit_cuts=getattr(args, "max_implicit_cuts", None),
        fragment_style=getattr(args, "fragment_style", None),
    )


def load_tokenizer(args, paths: OutputPaths, config: MolexarConfig) -> MolexarTokenizerFast:
    """Load an existing tokenizer, copying/training one if needed."""
    output_tokenizer = os.path.join(paths.tokenizer_dir, "tokenizer.json")
    candidates = [
        output_tokenizer,
        args.tokenizer_path,
        os.path.join(args.tokenizer_path, "tokenizer.json") if args.tokenizer_path else None,
        os.path.join(args.base_model, "tokenizer.json") if getattr(args, "base_model", None) else None,
        os.path.join(os.path.dirname(args.base_model), "tokenizer", "tokenizer.json")
        if getattr(args, "base_model", None)
        else None,
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            if candidate != output_tokenizer:
                os.makedirs(paths.tokenizer_dir, exist_ok=True)
                shutil.copy2(candidate, output_tokenizer)
                candidate = output_tokenizer
            return MolexarTokenizerFast(tokenizer_file=candidate)

    if args.task != "pretrain":
        raise FileNotFoundError("No tokenizer.json found. Pass --tokenizer_path or train a tokenizer first.")

    if not args.train_data_path:
        raise ValueError("--train_data_path is required to train a tokenizer")

    logger.warning("Tokenizer not found; training a tokenizer from the training data")
    train_file = args.train_data_path
    temp_file = None
    if args.data_format == "smiles":
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".fragment_selfies", delete=False)
        temp_file = handle.name
        with open(args.train_data_path, "r") as source:
            for line in source:
                parts = line.strip().split(maxsplit=1)
                if parts:
                    try:
                        handle.write(convert_smiles(parts[0], args) + "\n")
                    except Exception as exc:
                        logger.warning(f"Tokenizer conversion failed for {parts[0][:50]}: {exc}")
        handle.close()
        train_file = temp_file

    train_and_save_tokenizer([train_file], paths.tokenizer_dir, config)
    if temp_file:
        os.unlink(temp_file)
    return MolexarTokenizerFast(tokenizer_file=output_tokenizer)


class SMILESPretrainingDataset(Dataset):
    """Pretraining dataset that converts SMILES to Fragment-SELFIES on demand."""

    def __init__(self, file_path: str, tokenizer, config: MolexarConfig, args):
        self.items = []
        with open(file_path, "r") as handle:
            for line in handle:
                parts = line.strip().split(maxsplit=1)
                if parts:
                    self.items.append(parts[0])
        self.tokenizer = tokenizer
        self.config = config
        self.args = args
        self.cond_block, _ = build_condition_template(config)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        try:
            fragment_selfies = convert_smiles(self.items[index], self.args)
        except Exception as exc:
            logger.warning(f"SMILES conversion failed at row {index}: {exc}")
            fragment_selfies = ""
        text = build_training_text(self.config, self.cond_block, fragment_selfies)
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_position_embeddings,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": input_ids.clone(),
        }


class ConditionalSFTDataset(Dataset):
    """Generic SFT dataset for one or more condition keys."""

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        tokenizer,
        config: MolexarConfig,
        args,
        random_condition_subset: bool = False,
        min_condition_count: int = 1,
        max_condition_count: Optional[int] = None,
    ):
        self.records = list(records)
        self.tokenizer = tokenizer
        self.config = config
        self.args = args
        self.random_condition_subset = random_condition_subset
        self.min_condition_count = min_condition_count
        self.max_condition_count = max_condition_count
        self.cond_block, self.value_positions = build_condition_template(config)
        self.prefix_token_count = len(
            tokenizer.encode(
                f"{config.BOS_TOKEN}{self.cond_block}{config.SEP_TOKEN}{config.MOL_TOKEN}",
                add_special_tokens=False,
            )
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self._prepare_record(self.records[index], index)
        return self._encode_conditioned_molecule(
            record["molecule"],
            dict(record.get("conditions", {})),
            index,
            record.get("data_format", self.args.data_format),
        )

    def _encode_conditioned_molecule(
        self,
        molecule: str,
        conditions: Dict[str, Any],
        index: int,
        data_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        data_format = data_format or self.args.data_format
        if data_format == "smiles":
            try:
                molecule = convert_smiles(molecule, self.args)
            except Exception as exc:
                logger.warning(f"SMILES conversion failed at row {index}: {exc}")
                molecule = ""

        selected_keys = self._select_condition_keys(conditions)
        condition_values = {key: self._to_condition_value(conditions[key]) for key in selected_keys}
        condition_indices = {key: self.value_positions[key] for key in selected_keys}

        text = build_training_text(self.config, self.cond_block, molecule)
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_position_embeddings,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[: self.prefix_token_count] = -100
        if self.tokenizer.pad_token_id is not None:
            labels[input_ids == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
            "condition_values": condition_values,
            "condition_indices": condition_indices,
        }

    def _prepare_record(self, record: Dict[str, Any], index: int) -> Dict[str, Any]:
        return record

    def _select_condition_keys(self, conditions: Dict[str, Any]) -> List[str]:
        keys = [key for key in conditions if key in self.value_positions and conditions[key] is not None]
        if not self.random_condition_subset:
            return keys
        max_count = len(keys) if self.max_condition_count is None else self.max_condition_count
        max_count = min(max_count, len(keys))
        min_count = min(self.min_condition_count, max_count)
        if max_count == 0:
            return []
        count = random.randint(min_count, max_count)
        return random.sample(keys, count) if count else []

    def _to_condition_value(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor) or isinstance(value, dict):
            return value
        if isinstance(value, np.ndarray):
            return torch.tensor(value, dtype=torch.float32)
        if isinstance(value, (int, float)):
            return torch.tensor([value], dtype=torch.float32)
        if isinstance(value, list):
            return torch.tensor(value, dtype=torch.float32)
        return value


class MoleculePropertyConditionSFTDataset(ConditionalSFTDataset):
    """Aligned molecule-property SFT dataset with lazy molecule and property loading."""

    def __init__(self, molecule_path: str, properties_path: str, tokenizer, config: MolexarConfig, args):
        super().__init__(
            [],
            tokenizer,
            config,
            args,
            random_condition_subset=not args.disable_random_condition_subset,
            min_condition_count=args.min_condition_count,
            max_condition_count=args.max_condition_count,
        )
        offset_limit = args.max_samples if args.max_samples else None
        self.molecules = OffsetLineReader(molecule_path, max_lines=offset_limit)
        self.properties = OffsetLineReader(properties_path, skip_lines=1, max_lines=offset_limit)
        with open(properties_path, "r", newline="", encoding="utf-8") as handle:
            self.property_columns = [column.strip() for column in handle.readline().strip().split(",")]
        self.length = min(len(self.molecules), len(self.properties))
        if args.max_samples:
            self.length = min(self.length, args.max_samples)
        logger.info(
            "Initialized molecule property-condition SFT dataset: "
            f"samples={self.length}, molecule_path={molecule_path}"
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        molecule = self.molecules.line(index).split(maxsplit=1)[0]
        conditions = self._load_property_conditions(index)
        return self._encode_conditioned_molecule(molecule, conditions, index)

    def _load_property_conditions(self, index: int) -> Dict[str, Any]:
        values = next(csv.reader([self.properties.line(index)]))
        row = {key: value for key, value in zip(self.property_columns, values)}
        return _property_conditions_from_row(row)


class MoleculePharmaFPSFTDataset(ConditionalSFTDataset):
    """Aligned molecule-pharmacophore SFT dataset with lazy molecule and FP loading."""

    def __init__(self, molecule_path: str, fp_path: str, tokenizer, config: MolexarConfig, args):
        super().__init__([], tokenizer, config, args)
        offset_limit = args.max_samples if args.max_samples else None
        self.molecules = OffsetLineReader(molecule_path, max_lines=offset_limit)
        self.pharma_fps = np.load(fp_path, mmap_mode="r")
        self.pharma_fp_dim = config.condition_settings.get("mol_pharma_fp", {}).get("dim", 1032)
        self.length = min(len(self.molecules), self.pharma_fps.shape[0])
        if args.max_samples:
            self.length = min(self.length, args.max_samples)
        logger.info(
            "Initialized molecule pharmacophore-FP SFT dataset: "
            f"samples={self.length}, molecule_path={molecule_path}, pharma_fp={fp_path}"
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        molecule = self.molecules.line(index).split(maxsplit=1)[0]
        conditions = {"mol_pharma_fp": self._load_pharma_fp(index)}
        return self._encode_conditioned_molecule(molecule, conditions, index)

    def _load_pharma_fp(self, index: int) -> np.ndarray:
        return _load_pharma_fp_row(self.pharma_fps, index, self.pharma_fp_dim)


class MoleculeMultiConditionSFTDataset(ConditionalSFTDataset):
    """Aligned molecule-property-pharmacophore SFT dataset with lazy molecule loading."""

    def __init__(self, molecule_path: str, properties_path: str, fp_path: str, tokenizer, config: MolexarConfig, args):
        super().__init__(
            [],
            tokenizer,
            config,
            args,
            random_condition_subset=not args.disable_random_condition_subset,
            min_condition_count=args.min_condition_count,
            max_condition_count=args.max_condition_count,
        )
        offset_limit = args.max_samples if args.max_samples else None
        self.molecules = OffsetLineReader(molecule_path, max_lines=offset_limit)
        self.properties = OffsetLineReader(properties_path, skip_lines=1, max_lines=offset_limit)
        with open(properties_path, "r", newline="") as handle:
            self.property_columns = [column.strip() for column in handle.readline().strip().split(",")]
        self.pharma_fps = np.load(fp_path, mmap_mode="r")
        self.pharma_fp_dim = config.condition_settings.get("mol_pharma_fp", {}).get("dim", 1032)
        self.length = min(len(self.molecules), len(self.properties), self.pharma_fps.shape[0])
        if args.max_samples:
            self.length = min(self.length, args.max_samples)
        logger.info(
            "Initialized molecule multi-condition SFT dataset: "
            f"samples={self.length}, molecule_path={molecule_path}"
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        molecule = self.molecules.line(index).split(maxsplit=1)[0]
        conditions = self._load_property_conditions(index)
        conditions["mol_pharma_fp"] = self._load_pharma_fp(index)
        selected_keys = self._select_condition_keys(conditions)
        condition_values = {key: self._to_condition_value(conditions[key]) for key in selected_keys}
        condition_indices = {key: self.value_positions[key] for key in selected_keys}

        data_format = self.args.data_format
        if data_format == "smiles":
            try:
                molecule = convert_smiles(molecule, self.args)
            except Exception as exc:
                logger.warning(f"SMILES conversion failed at row {index}: {exc}")
                molecule = ""

        text = build_training_text(self.config, self.cond_block, molecule)
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_position_embeddings,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[: self.prefix_token_count] = -100
        if self.tokenizer.pad_token_id is not None:
            labels[input_ids == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
            "condition_values": condition_values,
            "condition_indices": condition_indices,
        }

    def _load_property_conditions(self, index: int) -> Dict[str, Any]:
        values = next(csv.reader([self.properties.line(index)]))
        row = {key: value for key, value in zip(self.property_columns, values)}
        return _property_conditions_from_row(row)

    def _load_pharma_fp(self, index: int) -> np.ndarray:
        return _load_pharma_fp_row(self.pharma_fps, index, self.pharma_fp_dim)


class FragmentSelfiesMoleculeContextSFTDataset(ConditionalSFTDataset):
    """UniChem molecule-context SFT dataset backed by randomized Fragment-SELFIES."""

    def __init__(self, fragment_selfies_path: str, properties_path: str, fp_path: str, tokenizer, config: MolexarConfig, args):
        super().__init__(
            [],
            tokenizer,
            config,
            args,
            random_condition_subset=not args.disable_random_condition_subset,
            min_condition_count=args.min_condition_count,
            max_condition_count=args.max_condition_count,
        )
        offset_limit = args.max_samples if args.max_samples else None
        self.fragments = OffsetLineReader(fragment_selfies_path, max_lines=offset_limit)
        self.properties = OffsetLineReader(properties_path, skip_lines=1, max_lines=offset_limit)
        with open(properties_path, "r", newline="", encoding="utf-8") as handle:
            self.property_columns = [column.strip() for column in handle.readline().strip().split(",")]
        self.pharma_fps = np.load(fp_path, mmap_mode="r")
        self.pharma_fp_dim = config.condition_settings.get("mol_pharma_fp", {}).get("dim", 1032)
        self.condition_length = min(len(self.properties), self.pharma_fps.shape[0])
        if self.condition_length <= 0:
            raise ValueError("Molecule context properties/pharmacophore files contain no usable rows")
        self.length = len(self.fragments)
        if args.max_samples:
            self.length = min(self.length, args.max_samples)
        self._check_fragment_condition_alignment(fragment_selfies_path, args)
        logger.info(
            "Initialized Fragment-SELFIES molecule-context SFT dataset: "
            f"samples={self.length}, condition_rows={self.condition_length}, "
            f"fragment_selfies={fragment_selfies_path}"
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        molecule = self.fragments.line(index)
        condition_index = index % self.condition_length
        conditions = self._load_property_conditions(condition_index)
        conditions["mol_pharma_fp"] = self._load_pharma_fp(condition_index)
        selected_keys = self._sample_molecule_context_keys(conditions)
        condition_values = {key: self._to_condition_value(conditions[key]) for key in selected_keys}
        condition_indices = {key: self.value_positions[key] for key in selected_keys}

        text = build_training_text(self.config, self.cond_block, molecule)
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_position_embeddings,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[: self.prefix_token_count] = -100
        if self.tokenizer.pad_token_id is not None:
            labels[input_ids == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
            "condition_values": condition_values,
            "condition_indices": condition_indices,
        }

    def _load_property_conditions(self, index: int) -> Dict[str, Any]:
        values = next(csv.reader([self.properties.line(index)]))
        row = {key: value for key, value in zip(self.property_columns, values)}
        return _property_conditions_from_row(row)

    def _load_pharma_fp(self, index: int) -> np.ndarray:
        return _load_pharma_fp_row(self.pharma_fps, index, self.pharma_fp_dim)

    def _sample_molecule_context_keys(self, conditions: Dict[str, Any]) -> List[str]:
        keys = [
            key for key in MOLECULE_CONDITION_KEYS
            if key in self.value_positions and key in conditions and conditions[key] is not None
        ]
        if not self.random_condition_subset:
            return keys[:3]
        return _sample_molecule_context_keys(keys, self.args, max_count=3)

    def _check_fragment_condition_alignment(self, fragment_selfies_path: str, args) -> None:
        if getattr(args, "skip_fragment_condition_alignment_check", False):
            return
        smiles_path = getattr(args, "smiles_file", None)
        if not smiles_path or not os.path.exists(smiles_path):
            return
        if os.path.abspath(smiles_path) == os.path.abspath(fragment_selfies_path):
            return
        if fragment_selfies_to_smiles is None:
            logger.warning("Skipping Fragment-SELFIES alignment check: converter is unavailable")
            return

        base_condition_indices = sorted({
            0,
            min(1, self.condition_length - 1),
            min(7, self.condition_length - 1),
            min(31, self.condition_length - 1),
        })
        fold_count = max(1, math.ceil(self.length / self.condition_length))
        sample_indices = []
        for fold_index in range(min(4, fold_count)):
            for condition_index in base_condition_indices:
                fragment_index = fold_index * self.condition_length + condition_index
                if fragment_index < self.length:
                    sample_indices.append(fragment_index)
        condition_indices = sorted({index % self.condition_length for index in sample_indices})
        expected_smiles = _read_first_column_at_indices(smiles_path, condition_indices)
        mismatches = []
        for fragment_index in sample_indices:
            condition_index = fragment_index % self.condition_length
            fragment = self.fragments.line(fragment_index)
            decoded = fragment_selfies_to_smiles(
                fragment,
                canonical=True,
                randomized=False,
                strict=False,
                ignore_errors=False,
            )
            expected = _canonicalize_smiles(expected_smiles[condition_index])
            if decoded != expected:
                mismatches.append((fragment_index, condition_index, decoded, expected))
        if mismatches:
            preview = "; ".join(
                f"fragment_row={item[0]} condition_row={item[1]} decoded={item[2]} expected={item[3]}"
                for item in mismatches[:3]
            )
            raise ValueError(f"Fragment-SELFIES condition alignment check failed: {preview}")
        logger.info(
            "Verified Fragment-SELFIES/condition alignment: "
            f"samples={len(sample_indices)}, condition_rows={self.condition_length}"
        )


class TargetContextSFTDataset(ConditionalSFTDataset):
    """SAIR/PLINDER target-context SFT dataset over protein-ligand pairs."""

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        molecule_condition_lookups: Dict[str, SmilesMoleculeConditionLookup],
        tokenizer,
        config: MolexarConfig,
        args,
    ):
        super().__init__(
            records,
            tokenizer,
            config,
            args,
            random_condition_subset=False,
            min_condition_count=args.min_condition_count,
            max_condition_count=args.max_condition_count,
        )
        self.molecule_condition_lookups = molecule_condition_lookups
        self.pocket_processor = None
        self.failed_pocket_paths = set()
        logger.info(f"Initialized target-context SFT dataset: samples={len(self.records)}")

    def __getitem__(self, index):
        errors = []
        for offset in range(len(self.records)):
            record = self.records[(index + offset) % len(self.records)]
            pair = record["pair"]
            try:
                molecule = self.molecule_condition_lookups[record["source"]].sample_fragment_selfies(
                    pair["smiles"]
                )
                conditions = self._sample_conditions(record, pair)
                return self._encode_sample(molecule, conditions)
            except Exception as exc:
                errors.append(str(exc))
                logger.warning(
                    f"Skipping failed target-context pair {record['source']} "
                    f"{record['target_id']} {pair.get('pocket_id')}: {exc}"
                )
        raise RuntimeError(
            "No usable target-context pair records remain: "
            f"{'; '.join(errors[:3])}"
        )

    def _encode_sample(self, molecule: str, conditions: Dict[str, Any]) -> Dict[str, Any]:
        condition_values = {key: self._to_condition_value(conditions[key]) for key in conditions}
        condition_indices = {key: self.value_positions[key] for key in conditions}

        text = build_training_text(self.config, self.cond_block, molecule)
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_position_embeddings,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[: self.prefix_token_count] = -100
        if self.tokenizer.pad_token_id is not None:
            labels[input_ids == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
            "condition_values": condition_values,
            "condition_indices": condition_indices,
        }

    def _sample_conditions(self, record: Dict[str, Any], pair: Dict[str, Any]) -> Dict[str, Any]:
        conditions = self._sample_target_conditions(record, pair)
        remaining_slots = 3 - len(conditions)
        if remaining_slots <= 0:
            return conditions

        ligand_probability = float(getattr(
            self.args,
            "target_ligand_condition_probability",
            DEFAULT_TARGET_LIGAND_CONDITION_PROBABILITY,
        ))
        if random.random() >= ligand_probability:
            return conditions

        molecule_conditions = self.molecule_condition_lookups[record["source"]].get(pair["smiles"])
        selected_molecule_keys = self._sample_molecule_condition_keys(molecule_conditions, remaining_slots)
        conditions.update({key: molecule_conditions[key] for key in selected_molecule_keys})
        return conditions

    def _sample_target_conditions(self, record: Dict[str, Any], pair: Dict[str, Any]) -> Dict[str, Any]:
        single_choices = []
        if record.get("esm_embedding") is not None:
            single_choices.append(("prot_seq_esm_emb",))
        if self._has_usable_pocket_path(pair):
            single_choices.append(("prot_poc_gvp_emb",))
        if not single_choices:
            raise ValueError(f"Target sample lacks both sequence and pocket context: {record['target_id']}")

        dual_choice = None
        if record.get("esm_embedding") is not None and self._has_usable_pocket_path(pair):
            dual_choice = ("prot_seq_esm_emb", "prot_poc_gvp_emb")
        dual_probability = float(getattr(
            self.args,
            "target_dual_context_probability",
            DEFAULT_TARGET_DUAL_CONTEXT_PROBABILITY,
        ))
        selected = dual_choice if dual_choice and random.random() < dual_probability else random.choice(single_choices)
        conditions: Dict[str, Any] = {}
        if "prot_seq_esm_emb" in selected:
            conditions["prot_seq_esm_emb"] = record["esm_embedding"]
        if "prot_poc_gvp_emb" in selected:
            conditions["prot_poc_gvp_emb"] = self._load_pocket(pair)
        return conditions

    def _load_pocket(self, pair: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        pocket_path = pair.get("pocket_path")
        if not pocket_path:
            raise ValueError(f"Missing pocket path for {pair.get('pocket_id')}")
        try:
            return self._get_pocket_processor().process_pdb(pocket_path)
        except Exception:
            self.failed_pocket_paths.add(pocket_path)
            raise

    def _get_pocket_processor(self) -> PocketProcessor:
        if self.pocket_processor is None:
            self.pocket_processor = PocketProcessor(
                pocket_radius=self.args.pocket_radius,
                max_atoms=self.args.max_atoms,
                include_hydrogens=False,
                node_scalar_dim=self.config.gvp_node_in_dim[0],
            )
        return self.pocket_processor

    def _has_usable_pocket_path(self, pair: Dict[str, Any]) -> bool:
        pocket_path = pair.get("pocket_path")
        return bool(pocket_path) and pocket_path not in self.failed_pocket_paths

    def _sample_molecule_condition_keys(self, conditions: Dict[str, Any], max_count: int) -> List[str]:
        keys = [
            key for key in MOLECULE_CONDITION_KEYS
            if key in self.value_positions and key in conditions and conditions[key] is not None
        ]
        if not keys:
            return []
        return _sample_condition_keys(keys, self.args, max_count=max_count)


class TargetContextSingleConditionSFTDataset(TargetContextSFTDataset):
    """SAIR/PLINDER target-context SFT dataset for one target condition key."""

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        molecule_condition_lookups: Dict[str, SmilesMoleculeConditionLookup],
        tokenizer,
        config: MolexarConfig,
        args,
        condition_key: str,
    ):
        if condition_key not in {"prot_seq_esm_emb", "prot_poc_gvp_emb"}:
            raise ValueError(f"Unsupported target-context condition key: {condition_key}")
        self.condition_key = condition_key
        super().__init__(records, molecule_condition_lookups, tokenizer, config, args)

    def _sample_conditions(self, record: Dict[str, Any], pair: Dict[str, Any]) -> Dict[str, Any]:
        if self.condition_key == "prot_seq_esm_emb":
            embedding = record.get("esm_embedding")
            if embedding is None:
                raise ValueError(f"Target sample lacks sequence context: {record['target_id']}")
            return {"prot_seq_esm_emb": embedding}

        if not self._has_usable_pocket_path(pair):
            raise ValueError(f"Target sample lacks pocket context: {record['target_id']}")
        return {"prot_poc_gvp_emb": self._load_pocket(pair)}


class UniversalRatioBatchSampler(Sampler[List[int]]):
    """Yield batches with a fixed molecule-context:target-context ratio."""

    def __init__(
        self,
        dataset: "UniversalMultiConditionSFTDataset",
        batch_size: int,
        drop_last: bool,
        seed: int,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        total_ratio = dataset.molecule_ratio + dataset.target_context_ratio
        target_count = int(round(batch_size * dataset.target_context_ratio / total_ratio))
        target_count = max(1, target_count) if dataset.target_context_len else 0
        if target_count >= batch_size and dataset.molecule_context_len:
            target_count = batch_size - 1
        self.target_per_batch = target_count
        self.molecule_per_batch = batch_size - target_count
        if self.molecule_per_batch <= 0:
            raise ValueError("Universal ratio batch sampler requires at least one molecule sample per batch")

    def __len__(self):
        if self.drop_last:
            return self.dataset.molecule_context_len // self.molecule_per_batch
        return math.ceil(self.dataset.molecule_context_len / self.molecule_per_batch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        target_order = list(range(self.dataset.target_context_offset, self.dataset.target_context_offset + self.dataset.target_context_len))
        rng.shuffle(target_order)
        target_cursor = 0

        def next_targets(count: int) -> List[int]:
            nonlocal target_cursor, target_order
            if count <= 0 or not target_order:
                return []
            selected = []
            while len(selected) < count:
                if target_cursor >= len(target_order):
                    rng.shuffle(target_order)
                    target_cursor = 0
                take = min(count - len(selected), len(target_order) - target_cursor)
                selected.extend(target_order[target_cursor:target_cursor + take])
                target_cursor += take
            return selected

        for start in range(0, self.dataset.molecule_context_len, self.molecule_per_batch):
            molecule_indices = list(range(start, min(start + self.molecule_per_batch, self.dataset.molecule_context_len)))
            if len(molecule_indices) < self.molecule_per_batch and self.drop_last:
                continue
            if len(molecule_indices) == self.molecule_per_batch:
                target_count = self.target_per_batch
            else:
                target_count = int(round(len(molecule_indices) * self.dataset.target_context_ratio / self.dataset.molecule_ratio))
                target_count = max(1, target_count) if self.dataset.target_context_len else 0
            batch = molecule_indices + next_targets(target_count)
            rng.shuffle(batch)
            yield batch


class UniversalMultiConditionSFTDataset(Dataset):
    """Universal SFT data over UniChem molecule context and SAIR/PLINDER target context."""

    def __init__(self, tokenizer, config: MolexarConfig, args):
        molecule_context_file = getattr(args, "molecule_context_file", None) or args.smiles_file
        self.molecule_context_dataset = FragmentSelfiesMoleculeContextSFTDataset(
            molecule_context_file,
            args.properties_file,
            args.pharma_fp_file,
            tokenizer,
            config,
            args,
        )
        target_records, molecule_condition_lookups = build_target_context_records(args, config)
        self.target_context_dataset = TargetContextSFTDataset(
            target_records,
            molecule_condition_lookups,
            tokenizer,
            config,
            args,
        )
        self.molecule_ratio = args.molecule_ratio
        self.target_context_ratio = args.target_context_ratio
        if min(self.molecule_ratio, self.target_context_ratio) < 0:
            raise ValueError("universal_multi ratios must be non-negative")
        if self.molecule_ratio <= 0:
            raise ValueError("universal_multi molecule_ratio must be positive")
        if self.target_context_ratio > 0 and len(self.target_context_dataset) == 0:
            raise ValueError("universal_multi target-context dataset must contain at least one sample")
        self.molecule_context_len = len(self.molecule_context_dataset)
        self.target_context_len = len(self.target_context_dataset)
        self.target_context_offset = self.molecule_context_len
        self.length = self.molecule_context_len + self.target_context_len
        logger.info(
            "Initialized universal multi-condition dataset: "
            f"length={self.length}, ratios={self.molecule_ratio}:{self.target_context_ratio}, "
            f"molecule_context={self.molecule_context_len}, "
            f"target_context={self.target_context_len}"
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index < self.molecule_context_len:
            return self.molecule_context_dataset[index]
        return self.target_context_dataset[index - self.target_context_offset]

    def build_batch_sampler(self, batch_size: int, drop_last: bool, seed: Optional[int]) -> UniversalRatioBatchSampler:
        return UniversalRatioBatchSampler(
            self,
            batch_size=batch_size,
            drop_last=drop_last,
            seed=0 if seed is None else seed,
        )


class ConditionalDataCollator:
    """Collate sparse per-sample condition dictionaries."""

    def __call__(self, features):
        batch = {
            "input_ids": torch.stack([feature["input_ids"] for feature in features]),
            "attention_mask": torch.stack([feature["attention_mask"] for feature in features]),
            "labels": torch.stack([feature["labels"] for feature in features]),
        }
        all_keys = set()
        for feature in features:
            all_keys.update(feature.get("condition_values", {}).keys())

        condition_values = {}
        condition_indices = {}
        for key in all_keys:
            present_values = [
                feature["condition_values"][key]
                for feature in features
                if key in feature.get("condition_values", {})
            ]
            template = present_values[0] if present_values else torch.zeros(1, dtype=torch.float32)
            values = []
            indices = []
            for feature in features:
                if key in feature.get("condition_values", {}):
                    values.append(feature["condition_values"][key])
                    indices.append(feature["condition_indices"][key])
                else:
                    values.append(self._missing_value(template))
                    indices.append(MISSING_CONDITION_INDEX)
            if isinstance(template, dict):
                condition_values[key] = values
            else:
                condition_values[key] = torch.stack([
                    value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.float32)
                    for value in values
                ])
            condition_indices[key] = indices

        batch["condition_values"] = condition_values
        batch["condition_indices"] = condition_indices
        return batch

    def _missing_value(self, template):
        if isinstance(template, dict):
            return None
        if isinstance(template, torch.Tensor):
            return torch.zeros_like(template)
        return torch.zeros(1, dtype=torch.float32)


class ConditionalTrainer(Trainer):
    """Trainer that forwards condition tensors and batches pocket graphs."""

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if not hasattr(self.train_dataset, "build_batch_sampler"):
            return super().get_train_dataloader()

        batch_sampler = self.train_dataset.build_batch_sampler(
            batch_size=self._train_batch_size,
            drop_last=getattr(self.args, "dataloader_drop_last", False),
            seed=getattr(self.args, "seed", None),
        )
        dataloader_params = {
            "batch_sampler": batch_sampler,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "worker_init_fn": _seed_dataloader_worker,
        }
        if self.args.dataloader_num_workers > 0:
            dataloader_params["persistent_workers"] = self.args.dataloader_persistent_workers
            prefetch_factor = getattr(self.args, "dataloader_prefetch_factor", None)
            if prefetch_factor is not None:
                dataloader_params["prefetch_factor"] = prefetch_factor
        return self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        condition_values = inputs.pop("condition_values", None)
        condition_indices = inputs.pop("condition_indices", None)
        condition_values = self._batch_pocket_conditions(model, condition_values)
        outputs = model(**inputs, condition_values=condition_values, condition_indices=condition_indices)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def _batch_pocket_conditions(self, model, condition_values):
        if not condition_values or "prot_poc_gvp_emb" not in condition_values:
            return condition_values
        pocket_values = condition_values["prot_poc_gvp_emb"]
        if not isinstance(pocket_values, list) or not pocket_values:
            return condition_values
        if not any(isinstance(pocket, dict) for pocket in pocket_values):
            return condition_values

        unwrapped = model.module if hasattr(model, "module") else model
        device = next(unwrapped.parameters()).device
        node_scalar, node_vector, edge_index, edge_scalar, edge_vector, batch_idx = [], [], [], [], [], []
        node_offset = 0
        for index, pocket in enumerate(pocket_values):
            if pocket is None:
                pocket = {
                    "node_scalar": torch.zeros(1, unwrapped.config.gvp_node_in_dim[0], device=device),
                    "node_vector": torch.zeros(1, unwrapped.config.gvp_node_in_dim[1], 3, device=device),
                    "edge_index": torch.zeros(2, 0, dtype=torch.long, device=device),
                    "edge_scalar": torch.zeros(0, unwrapped.config.gvp_edge_in_dim[0], device=device),
                    "edge_vector": torch.zeros(0, unwrapped.config.gvp_edge_in_dim[1], 3, device=device),
                }
            n_nodes = pocket["node_scalar"].shape[0]
            node_scalar.append(pocket["node_scalar"].to(device))
            node_vector.append(pocket["node_vector"].to(device))
            edge_index.append(pocket["edge_index"].to(device) + node_offset)
            edge_scalar.append(pocket["edge_scalar"].to(device))
            edge_vector.append(pocket["edge_vector"].to(device))
            batch_idx.append(torch.full((n_nodes,), index, dtype=torch.long, device=device))
            node_offset += n_nodes

        condition_values = dict(condition_values)
        condition_values["prot_poc_gvp_emb"] = {
            "node_scalar": torch.cat(node_scalar, dim=0),
            "node_vector": torch.cat(node_vector, dim=0),
            "edge_index": torch.cat(edge_index, dim=1),
            "edge_scalar": torch.cat(edge_scalar, dim=0),
            "edge_vector": torch.cat(edge_vector, dim=0),
            "batch": torch.cat(batch_idx, dim=0),
        }
        return condition_values


class MetricsLoggerCallback(TrainerCallback):
    """Write basic training metrics to JSONL."""

    def __init__(self, metrics_dir: str):
        self.metrics_path = os.path.join(metrics_dir, "training_metrics.jsonl")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        row = {"step": state.global_step, **logs}
        with open(self.metrics_path, "a") as handle:
            handle.write(json.dumps(row) + "\n")


def load_model(args, config: MolexarConfig) -> MolexarForCausalLM:
    """Load or initialize a model for pretraining/SFT."""
    if args.task == "pretrain":
        model = MolexarForCausalLM(config)
        for parameter in model.model.condition_encoders.parameters():
            parameter.requires_grad = False
        for parameter in model.model.gvp_encoder.parameters():
            parameter.requires_grad = False
        return model

    if not args.base_model:
        raise ValueError("--base_model is required for SFT")
    model = MolexarForCausalLM(config)
    state_dict = _load_state_dict(args.base_model)
    excluded_prefixes = ("model.gvp_encoder.",) if args.sft_mode == "universal_multi" else (
        "model.condition_encoders.",
        "model.gvp_encoder.",
    )
    filtered = {key: value for key, value in state_dict.items() if not key.startswith(excluded_prefixes)}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    non_condition_missing = [key for key in missing if not key.startswith(excluded_prefixes)]
    if non_condition_missing:
        logger.warning(f"Unexpected missing keys: {non_condition_missing[:10]}")
    if unexpected:
        logger.warning(f"Unexpected checkpoint keys: {unexpected[:10]}")
    return model


def _load_state_dict(model_dir: str) -> Dict[str, torch.Tensor]:
    candidates = [
        os.path.join(model_dir, "pytorch_model_fsdp.bin"),
        os.path.join(model_dir, "pytorch_model.bin"),
        os.path.join(model_dir, "model.safetensors"),
        os.path.join(model_dir, "pytorch_model_safe.bin"),
    ]
    for path in candidates:
        if os.path.exists(path):
            logger.info(f"Loading weights from {path}")
            if path.endswith(".safetensors"):
                from safetensors import safe_open

                with safe_open(path, framework="pt") as handle:
                    return {key: handle.get_tensor(key) for key in handle.keys()}
            return torch.load(path, map_location="cpu")
    raise FileNotFoundError(f"No model weights found in {model_dir}")


def build_datasets(args, tokenizer, config: MolexarConfig) -> Tuple[Dataset, Optional[Dataset], bool]:
    """Build train/eval datasets and return whether the conditional trainer is needed."""
    if args.task == "pretrain":
        train = _build_pretrain_dataset(args.train_data_path, tokenizer, config, args)
        eval_dataset = _build_pretrain_dataset(args.eval_data_path, tokenizer, config, args) if args.eval_data_path else None
        return train, eval_dataset, False

    train = build_sft_dataset(args, tokenizer, config, validation=False)
    eval_dataset = build_sft_dataset(args, tokenizer, config, validation=True)
    return train, eval_dataset, True


def _build_pretrain_dataset(path: str, tokenizer, config: MolexarConfig, args) -> Dataset:
    if not path:
        raise ValueError("--train_data_path is required for pretraining")
    if args.data_format == "smiles":
        return SMILESPretrainingDataset(path, tokenizer, config, args)
    return PretrainingDataset(path, tokenizer, config, max_sequence_length=config.max_position_embeddings)


def build_sft_dataset(args, tokenizer, config: MolexarConfig, validation: bool = False) -> Optional[Dataset]:
    """Build one SFT dataset for the selected data mode."""
    if args.sft_mode == "molecule_multi":
        if validation and not (args.val_smiles_file and args.val_properties_file and args.val_pharma_fp_file):
            return None
        return build_molecule_multi_dataset(args, tokenizer, config, validation=validation)
    if args.sft_mode == "universal_multi":
        if validation:
            return None
        return UniversalMultiConditionSFTDataset(tokenizer, config, args)
    if args.sft_mode == "generic":
        path = args.val_sft_data_path if validation else args.sft_data_path
        if validation and not path:
            return None
        records = load_generic_records(path)
        return ConditionalSFTDataset(records, tokenizer, config, args)
    if args.sft_mode == "properties":
        mol_path = args.val_smiles_file if validation else args.smiles_file
        prop_path = args.val_properties_file if validation else args.properties_file
        if validation and not (mol_path and prop_path):
            return None
        return MoleculePropertyConditionSFTDataset(mol_path, prop_path, tokenizer, config, args)
    if args.sft_mode == "pharma_fp":
        mol_path = args.val_smiles_file if validation else args.smiles_file
        fp_path = args.val_pharma_fp_file if validation else args.pharma_fp_file
        if validation and not (mol_path and fp_path):
            return None
        return MoleculePharmaFPSFTDataset(mol_path, fp_path, tokenizer, config, args)
    if args.sft_mode == "protein_sequence":
        if validation:
            return None
        return build_target_context_single_condition_dataset(
            args,
            tokenizer,
            config,
            "prot_seq_esm_emb",
        )
    if args.sft_mode == "protein_pocket":
        if validation:
            return None
        return build_target_context_single_condition_dataset(
            args,
            tokenizer,
            config,
            "prot_poc_gvp_emb",
        )
    raise ValueError(f"Unknown SFT mode: {args.sft_mode}")


def build_molecule_multi_dataset(args, tokenizer, config: MolexarConfig, validation: bool) -> MoleculeMultiConditionSFTDataset:
    molecule_path = args.val_smiles_file if validation else args.smiles_file
    properties_path = args.val_properties_file if validation else args.properties_file
    pharma_fp_path = args.val_pharma_fp_file if validation else args.pharma_fp_file
    if not (molecule_path and properties_path and pharma_fp_path):
        raise ValueError("--smiles_file, --properties_file, and --pharma_fp_file are required")
    return MoleculeMultiConditionSFTDataset(
        molecule_path,
        properties_path,
        pharma_fp_path,
        tokenizer,
        config,
        args,
    )


def build_target_context_records(args, config: MolexarConfig) -> Tuple[List[Dict[str, Any]], Dict[str, SmilesMoleculeConditionLookup]]:
    if args.sft_mode == "universal_multi" and getattr(args, "target_context_ratio", 1) <= 0:
        return [], {}

    sources = _parse_target_context_sources(getattr(args, "target_context_sources", "sair,plinder"))
    records: List[Dict[str, Any]] = []
    molecule_condition_lookups: Dict[str, SmilesMoleculeConditionLookup] = {}
    max_records_per_source = math.ceil(args.max_samples / len(sources)) if args.max_samples else None
    cache_dir = getattr(args, "target_context_cache_dir", None)
    verify_pocket_paths = getattr(args, "verify_target_pocket_paths", True)
    fragment_selfies_folds = getattr(args, "target_ligand_fragment_selfies_folds", 10)

    if "sair" in sources:
        sair_index_dir = getattr(args, "sair_index_dir", None)
        sair_structures_dir = getattr(args, "sair_structures_dir", None)
        sair_smiles_path = os.path.join(
            sair_index_dir,
            "unique_ligands.no_hydrogen.no_stereo.deduplicated.filtered.smi",
        )
        molecule_condition_lookups["sair"] = SmilesMoleculeConditionLookup(
            sair_smiles_path,
            os.path.join(sair_index_dir, "unique_ligands.no_hydrogen.no_stereo.deduplicated.filtered.properties.csv"),
            os.path.join(sair_index_dir, "unique_ligands.no_hydrogen.no_stereo.deduplicated.filtered.gobbi_pharma_fps.npy"),
            config,
            "SAIR",
            _target_ligand_fragment_selfies_path(args, "sair", sair_smiles_path),
            fragment_selfies_folds,
        )
        records.extend(
            load_sair_target_context_records(
                sair_index_dir,
                sair_structures_dir,
                config,
                max_records=max_records_per_source,
                cache_dir=cache_dir,
                verify_pocket_paths=verify_pocket_paths,
                pairs_path=getattr(args, "sair_pairs_file", None),
            )
        )

    if "plinder" in sources:
        plinder_index_dir = getattr(args, "plinder_index_dir", None) or os.path.join(args.plinder_root, "index")
        plinder_smiles_path = os.path.join(
            plinder_index_dir,
            "unique_ligands.no_hydrogen.no_stereo.deduplicated.filtered.smi",
        )
        molecule_condition_lookups["plinder"] = SmilesMoleculeConditionLookup(
            plinder_smiles_path,
            os.path.join(plinder_index_dir, "unique_ligands.no_hydrogen.no_stereo.deduplicated.filtered.properties.csv"),
            os.path.join(plinder_index_dir, "unique_ligands.no_hydrogen.no_stereo.deduplicated.filtered.gobbi_pharma_fps.npy"),
            config,
            "PLINDER",
            _target_ligand_fragment_selfies_path(args, "plinder", plinder_smiles_path),
            fragment_selfies_folds,
        )
        records.extend(
            load_plinder_target_context_records(
                plinder_index_dir,
                args.plinder_root,
                config,
                max_records=max_records_per_source,
                cache_dir=cache_dir,
                verify_pocket_paths=verify_pocket_paths,
                pairs_path=getattr(args, "plinder_pairs_file", None),
            )
        )

    if args.max_samples:
        records = records[: args.max_samples]
    return records, molecule_condition_lookups


def build_target_context_single_condition_dataset(
    args,
    tokenizer,
    config: MolexarConfig,
    condition_key: str,
) -> TargetContextSingleConditionSFTDataset:
    records, molecule_condition_lookups = build_target_context_records(args, config)
    if condition_key == "prot_seq_esm_emb":
        records = [record for record in records if record.get("esm_embedding") is not None]
    elif condition_key == "prot_poc_gvp_emb":
        records = [record for record in records if (record.get("pair") or {}).get("pocket_path")]
    else:
        raise ValueError(f"Unsupported target-context condition key: {condition_key}")
    if not records:
        raise ValueError(f"Target-context dataset has no usable {condition_key} samples")
    return TargetContextSingleConditionSFTDataset(
        records,
        molecule_condition_lookups,
        tokenizer,
        config,
        args,
        condition_key,
    )


def _target_ligand_fragment_selfies_path(args, source: str, smiles_path: str) -> str:
    explicit_path = getattr(args, f"{source}_ligand_fragment_selfies_file", None)
    if explicit_path:
        return explicit_path

    if smiles_path.endswith(".smi"):
        base_path = smiles_path[:-4]
    else:
        base_path = smiles_path
    candidates = [
        f"{base_path}.randomized.10_fold.fragment_selfies",
        f"{base_path}.10_fold.fragment_selfies",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def load_sair_target_context_records(
    index_dir: str,
    structures_dir: str,
    config: MolexarConfig,
    max_records: Optional[int] = None,
    cache_dir: Optional[str] = None,
    verify_pocket_paths: bool = False,
    pairs_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    proteins_path = os.path.join(index_dir, "unique_proteins.filtered.csv")
    embeddings_path = os.path.join(index_dir, "unique_proteins.filtered.esmc_600m_embeddings.pkl")
    if pairs_path is None:
        pairs_path = os.path.join(
            index_dir,
            "unique_protein_ligand_pairs.no_hydrogen.no_stereo.filtered.deduplicated.csv",
        )
    for label, path in (("SAIR proteins", proteins_path), ("SAIR ESM embeddings", embeddings_path), ("SAIR pairs", pairs_path)):
        _require_file(path, label)

    expected_dim = config.condition_settings.get("prot_seq_esm_emb", {}).get("dim")
    metadata = _target_context_cache_metadata(
        "sair",
        max_records,
        expected_dim,
        verify_pocket_paths,
        [proteins_path, embeddings_path, pairs_path],
        {"structures_dir": structures_dir},
    )
    return _load_or_build_target_context_cache(
        cache_dir or os.path.join(index_dir, ".molexar_cache"),
        "sair",
        metadata,
        lambda: _load_sair_target_context_records_uncached(
            proteins_path,
            embeddings_path,
            pairs_path,
            structures_dir,
            expected_dim,
            max_records,
            verify_pocket_paths,
        ),
    )


def _load_sair_target_context_records_uncached(
    proteins_path: str,
    embeddings_path: str,
    pairs_path: str,
    structures_dir: str,
    expected_dim: Optional[int],
    max_records: Optional[int],
    verify_pocket_paths: bool,
) -> List[Dict[str, Any]]:
    embeddings = _load_pickle_tensor_map(embeddings_path, expected_dim=expected_dim)
    proteins: Dict[str, Dict[str, Any]] = {}
    with open(proteins_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            uniprot_id = (row.get("uniprot_id") or "").strip()
            if not uniprot_id:
                continue
            proteins[uniprot_id] = {"esm_embedding": embeddings.get(uniprot_id)}

    records: List[Dict[str, Any]] = []
    with open(pairs_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            uniprot_id = (row.get("uniprot_id") or "").strip()
            smiles = (row.get("smiles") or "").strip()
            entry_id = (row.get("entry_id") or "").strip()
            protein = proteins.get(uniprot_id)
            if protein is None or not smiles or not entry_id:
                continue
            pair = {
                "smiles": smiles,
                "pocket_id": entry_id,
                "pocket_path": _resolve_sair_pocket_path(
                    structures_dir,
                    entry_id,
                    verify=verify_pocket_paths,
                ),
            }
            record = {
                "source": "sair",
                "target_id": uniprot_id,
                "esm_embedding": protein["esm_embedding"],
                "pair": pair,
            }
            if _is_usable_target_pair_record(record):
                records.append(record)
                if max_records and len(records) >= max_records:
                    break

    target_count = len({record["target_id"] for record in records})
    pocket_count = sum(1 for record in records if record["pair"].get("pocket_path"))
    logger.info(
        "Loaded SAIR target-context records: "
        f"pair_records={len(records)}, proteins={target_count}, embeddings={len(embeddings)}, "
        f"pairs_with_pocket_path={pocket_count}, missing_pocket_paths={len(records) - pocket_count}"
    )
    return records


def load_plinder_target_context_records(
    index_dir: str,
    plinder_root: str,
    config: MolexarConfig,
    max_records: Optional[int] = None,
    cache_dir: Optional[str] = None,
    verify_pocket_paths: bool = False,
    pairs_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    proteins_path = os.path.join(index_dir, "unique_proteins.filtered.csv")
    embeddings_path = os.path.join(index_dir, "unique_proteins.filtered.esmc_600m_embeddings.pkl")
    if pairs_path is None:
        pairs_path = os.path.join(
            index_dir,
            "unique_protein_ligand_pairs.no_hydrogen.no_stereo.filtered.with_protein_id.csv",
        )
    for label, path in (("PLINDER proteins", proteins_path), ("PLINDER ESM embeddings", embeddings_path), ("PLINDER pairs", pairs_path)):
        _require_file(path, label)

    expected_dim = config.condition_settings.get("prot_seq_esm_emb", {}).get("dim")
    metadata = _target_context_cache_metadata(
        "plinder",
        max_records,
        expected_dim,
        verify_pocket_paths,
        [proteins_path, embeddings_path, pairs_path],
        {"plinder_root": plinder_root},
    )
    return _load_or_build_target_context_cache(
        cache_dir or os.path.join(index_dir, ".molexar_cache"),
        "plinder",
        metadata,
        lambda: _load_plinder_target_context_records_uncached(
            proteins_path,
            embeddings_path,
            pairs_path,
            plinder_root,
            expected_dim,
            max_records,
            verify_pocket_paths,
        ),
    )


def _load_plinder_target_context_records_uncached(
    proteins_path: str,
    embeddings_path: str,
    pairs_path: str,
    plinder_root: str,
    expected_dim: Optional[int],
    max_records: Optional[int],
    verify_pocket_paths: bool,
) -> List[Dict[str, Any]]:
    embeddings = _load_pickle_tensor_map(embeddings_path, expected_dim=expected_dim)
    proteins: Dict[str, Dict[str, Any]] = {}
    with open(proteins_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            protein_id = (row.get("protein_id") or "").strip()
            if not protein_id:
                continue
            proteins[protein_id] = {
                "esm_embedding": _get_embedding_by_text_or_int_key(embeddings, protein_id)
            }

    records: List[Dict[str, Any]] = []
    with open(pairs_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            protein_id = (row.get("protein_id") or "").strip()
            smiles = (row.get("smiles") or "").strip()
            system_id = (row.get("system_id") or "").strip()
            protein = proteins.get(protein_id)
            if protein is None or not smiles or not system_id:
                continue
            pair = {
                "smiles": smiles,
                "pocket_id": system_id,
                "pocket_path": _resolve_plinder_pocket_path(
                    plinder_root,
                    system_id,
                    verify=verify_pocket_paths,
                ),
            }
            record = {
                "source": "plinder",
                "target_id": protein_id,
                "esm_embedding": protein["esm_embedding"],
                "pair": pair,
            }
            if _is_usable_target_pair_record(record):
                records.append(record)
                if max_records and len(records) >= max_records:
                    break

    target_count = len({record["target_id"] for record in records})
    pocket_count = sum(1 for record in records if record["pair"].get("pocket_path"))
    logger.info(
        "Loaded PLINDER target-context records: "
        f"pair_records={len(records)}, proteins={target_count}, embeddings={len(embeddings)}, "
        f"pairs_with_pocket_path={pocket_count}, missing_pocket_paths={len(records) - pocket_count}"
    )
    return records


def _parse_target_context_sources(value: Any) -> List[str]:
    if isinstance(value, str):
        sources = [item.strip().lower() for item in value.split(",") if item.strip()]
    else:
        sources = [str(item).strip().lower() for item in value]
    invalid = sorted(set(sources) - {"sair", "plinder"})
    if invalid:
        raise ValueError(f"Unknown target context source(s): {', '.join(invalid)}")
    if not sources:
        raise ValueError("At least one target context source is required")
    return sources


def _is_usable_target_pair_record(record: Dict[str, Any]) -> bool:
    pair = record.get("pair") or {}
    return bool(pair.get("smiles")) and (
        record.get("esm_embedding") is not None or bool(pair.get("pocket_path"))
    )


def _resolve_sair_pocket_path(structures_dir: str, entry_id: str, verify: bool = True) -> Optional[str]:
    return resolve_sair_pocket_path(structures_dir, entry_id, verify=verify)


def _resolve_plinder_pocket_path(plinder_root: str, system_id: str, verify: bool = True) -> Optional[str]:
    return resolve_plinder_pocket_path(plinder_root, system_id, verify=verify)


def _target_context_cache_metadata(
    source: str,
    max_records: Optional[int],
    expected_dim: Optional[int],
    verify_pocket_paths: bool,
    input_paths: Sequence[str],
    roots: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "version": 4,
        "source": source,
        "max_records": max_records,
        "expected_dim": expected_dim,
        "verify_pocket_paths": verify_pocket_paths,
        "inputs": [_file_signature(path) for path in input_paths],
        "roots": roots,
    }


def _file_signature(path: str) -> Dict[str, Any]:
    stat_result = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
    }


def _load_or_build_target_context_cache(
    cache_dir: Optional[str],
    source: str,
    metadata: Dict[str, Any],
    builder,
) -> List[Dict[str, Any]]:
    if not cache_dir:
        return builder()
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as exc:
        logger.warning(f"Could not create target-context cache directory {cache_dir}: {exc}")
        return builder()

    cache_key = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"{source}_target_context_records_{cache_key}.pkl")
    lock_path = f"{cache_path}.lock"
    cached = _read_target_context_cache(cache_path, metadata)
    if cached is not None:
        logger.info(f"Loaded {source.upper()} target-context cache: {cache_path}")
        return cached

    while True:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            cached = _read_target_context_cache(cache_path, metadata)
            if cached is not None:
                logger.info(f"Loaded {source.upper()} target-context cache: {cache_path}")
                return cached
            try:
                lock_age = time.time() - os.stat(lock_path).st_mtime
            except OSError:
                lock_age = 0.0
            if lock_age > 6 * 60 * 60:
                try:
                    os.unlink(lock_path)
                except OSError:
                    pass
                continue
            time.sleep(2.0)
            continue
        except OSError as exc:
            logger.warning(f"Could not lock target-context cache {cache_path}: {exc}")
            return builder()
        break

    try:
        with os.fdopen(lock_fd, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()} time={time.time()}\n")
        cached = _read_target_context_cache(cache_path, metadata)
        if cached is not None:
            logger.info(f"Loaded {source.upper()} target-context cache: {cache_path}")
            return cached
        records = builder()
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=cache_dir,
                prefix=f".{os.path.basename(cache_path)}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = handle.name
                pickle.dump({"metadata": metadata, "records": records}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_path, cache_path)
            logger.info(f"Saved {source.upper()} target-context cache: {cache_path}")
        except OSError as exc:
            logger.warning(f"Could not save target-context cache {cache_path}: {exc}")
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
        return records
    finally:
        try:
            os.unlink(lock_path)
        except OSError:
            pass


def _read_target_context_cache(cache_path: str, metadata: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as handle:
            payload = pickle.load(handle)
    except (EOFError, OSError, pickle.PickleError, AttributeError, ValueError) as exc:
        logger.warning(f"Ignoring invalid target-context cache {cache_path}: {exc}")
        return None
    if not isinstance(payload, dict) or payload.get("metadata") != metadata:
        return None
    records = payload.get("records")
    if not isinstance(records, list):
        return None
    return records


def _get_embedding_by_text_or_int_key(embeddings: Dict[Any, torch.Tensor], key: str) -> Optional[torch.Tensor]:
    if key in embeddings:
        return embeddings[key]
    try:
        int_key = int(key)
    except ValueError:
        return None
    return embeddings.get(int_key)


def _require_file(path: str, label: str) -> None:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")


def load_generic_records(path: str) -> List[Dict[str, Any]]:
    if not path:
        raise ValueError("--sft_data_path is required for generic SFT")
    records = []
    if path.endswith(".jsonl"):
        with open(path, "r") as handle:
            for line in handle:
                if line.strip():
                    records.append(_normalize_generic_record(json.loads(line)))
    else:
        with open(path, "r") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and "records" in data:
            data = data["records"]
        records = [_normalize_generic_record(item) for item in data]
    return records


def _normalize_generic_record(item: Dict[str, Any]) -> Dict[str, Any]:
    molecule = item.get("fragment_selfies") or item.get("target") or item.get("molecule") or item.get("smiles")
    if molecule is None:
        raise ValueError("Generic SFT records need fragment_selfies, target, molecule, or smiles")
    data_format = item.get("data_format", "smiles" if "smiles" in item else "fragment_selfies")
    return {"molecule": molecule, "data_format": data_format, "conditions": item.get("conditions", {})}


def load_property_records(molecule_path: str, properties_path: str, args) -> List[Dict[str, Any]]:
    molecules = _load_first_column_lines(molecule_path)
    properties = []
    with open(properties_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            properties.append(_property_conditions_from_row(row))
    if len(molecules) != len(properties):
        raise ValueError("Molecule and property files must be aligned by line count")
    records = [
        {"molecule": molecule, "data_format": args.data_format, "conditions": props}
        for molecule, props in zip(molecules, properties)
    ]
    return records[: args.max_samples] if args.max_samples else records


def load_pharma_records(molecule_path: str, fp_path: str, config: MolexarConfig, args) -> List[Dict[str, Any]]:
    molecules = _load_first_column_lines(molecule_path)
    packed = np.load(fp_path)
    dim = config.condition_settings.get("mol_pharma_fp", {}).get("dim", 1032)
    if packed.shape[-1] == dim:
        fps = packed.astype(np.float32)
    else:
        fps = np.unpackbits(packed, axis=1, count=dim).astype(np.float32)
    if len(molecules) != len(fps):
        raise ValueError("Molecule and pharmacophore fingerprint files must be aligned by line count")
    records = [
        {"molecule": molecule, "data_format": args.data_format, "conditions": {"mol_pharma_fp": fp.astype(np.float32)}}
        for molecule, fp in zip(molecules, fps)
    ]
    return records[: args.max_samples] if args.max_samples else records


def _load_first_column_lines(path: str) -> List[str]:
    with open(path, "r") as handle:
        return [line.strip().split()[0] for line in handle if line.strip()]


def _read_first_column_at_indices(path: str, indices: Sequence[int]) -> Dict[int, str]:
    wanted = set(indices)
    found: Dict[int, str] = {}
    if not wanted:
        return found
    max_index = max(wanted)
    row_index = 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            if row_index in wanted:
                found[row_index] = line.strip().split(maxsplit=1)[0]
                if len(found) == len(wanted):
                    break
            row_index += 1
            if row_index > max_index and len(found) == len(wanted):
                break
    missing = sorted(wanted - set(found))
    if missing:
        raise ValueError(f"Could not read SMILES rows {missing[:5]} from {path}")
    return found


def _canonicalize_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem
    except ImportError:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def _property_conditions_from_row(row: Dict[str, str]) -> Dict[str, Any]:
    props = {}
    for key in PROPERTY_KEYS:
        value = float(_get_property_cell(row, PROPERTY_COLUMN_MAP[key]))
        props[key] = int(value) if key in DISCRETE_PROPERTY_KEYS else value
    return props


def _load_pharma_fp_row(pharma_fps: np.ndarray, index: int, pharma_fp_dim: int) -> np.ndarray:
    row = np.asarray(pharma_fps[index])
    if row.shape[-1] == pharma_fp_dim:
        return row.astype(np.float32)
    return np.unpackbits(row.astype(np.uint8), count=pharma_fp_dim).astype(np.float32)


def _get_identifier_cell(row: Dict[str, str]) -> Optional[str]:
    for candidate in ("mol_id", "molecule_id", "molecule_chembl_id", "system_id", "id", "ID"):
        value = row.get(candidate)
        if value:
            return value.strip()
    return None


def _is_valid_condition_row(row: Dict[str, str]) -> bool:
    value = row.get("valid") or row.get("VALID")
    if value is None or value == "":
        return True
    return value.strip().lower() not in {"0", "false", "no"}


def _get_property_cell(row: Dict[str, str], column: str) -> str:
    for candidate in (column, column.lower(), column.upper()):
        if candidate in row:
            return row[candidate]
    raise KeyError(f"Missing property column: {column}")


def _load_pickle_tensor_map(
    path: str,
    expected_dim: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    result = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            tensor = value.float()
        elif isinstance(value, np.ndarray):
            tensor = torch.tensor(value, dtype=torch.float32)
        elif isinstance(value, list):
            tensor = torch.tensor(value, dtype=torch.float32)
        elif hasattr(value, "numpy"):
            tensor = torch.tensor(value.numpy(), dtype=torch.float32)
        else:
            continue
        if tensor.dim() != 1:
            logger.warning(
                f"Skipping ESM embedding for {key}: "
                f"expected 1D tensor, got {tuple(tensor.shape)}"
            )
            continue
        if expected_dim is not None and tensor.shape[0] != expected_dim:
            logger.warning(
                f"Skipping ESM embedding for {key}: "
                f"expected dimension {expected_dim}, got {tensor.shape[0]}"
            )
            continue
        result[key] = tensor
    return result


def build_training_arguments(args, paths: OutputPaths, has_eval: bool) -> TrainingArguments:
    training_args = {
        "output_dir": paths.checkpoint_dir,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "save_total_limit": args.save_total_limit,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_pin_memory": True,
        "dataloader_num_workers": args.dataloader_num_workers,
        "seed": args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little"),
    }
    if has_eval:
        training_args.update({"eval_strategy": "steps", "eval_steps": args.eval_steps, "load_best_model_at_end": False})
    if args.use_fsdp:
        training_args["fsdp"] = f"{args.fsdp_strategy} auto_wrap"
        training_args["fsdp_config"] = {
            "backward_prefetch": "BACKWARD_PRE",
            "cpu_ram_efficient_loading": True,
            "forward_prefetch": False,
            "sync_module_states": True,
            "use_orig_params": True,
        }
    return TrainingArguments(**training_args)
