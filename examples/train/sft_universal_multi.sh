#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MODEL_NAME="${MODEL_NAME:-10m_256h_16l}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-${PROJECT_DIR}/outputs/pretrain_base_${MODEL_NAME}/final_model}"
MOLECULE_CONTEXT_PATH="${MOLECULE_CONTEXT_PATH:-/path/to/molecule_context.fragment_selfies}"
PROPERTIES_PATH="${PROPERTIES_PATH:-/path/to/molecule_properties.csv}"
PHARMA_FP_PATH="${PHARMA_FP_PATH:-/path/to/gobbi_pharma_fps.npy}"
SAIR_INDEX_DIR="${SAIR_INDEX_DIR:-/path/to/SAIR/index}"
SAIR_STRUCTURES_DIR="${SAIR_STRUCTURES_DIR:-/path/to/SAIR/structures_processed}"
SAIR_LIGAND_FRAGMENT_SELFIES_PATH="${SAIR_LIGAND_FRAGMENT_SELFIES_PATH:-/path/to/SAIR/ligands.fragment_selfies}"
PLINDER_ROOT="${PLINDER_ROOT:-/path/to/PLINDER/2024-06/v2}"
PLINDER_INDEX_DIR="${PLINDER_INDEX_DIR:-${PLINDER_ROOT}/index}"
PLINDER_LIGAND_FRAGMENT_SELFIES_PATH="${PLINDER_LIGAND_FRAGMENT_SELFIES_PATH:-/path/to/PLINDER/ligands.fragment_selfies}"
TARGET_CONTEXT_SOURCES="${TARGET_CONTEXT_SOURCES:-sair,plinder}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${PROJECT_DIR}/models/tokenizer}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/sft_universal_multi_${MODEL_NAME}}"
TARGET_CONTEXT_CACHE_DIR="${TARGET_CONTEXT_CACHE_DIR:-${PROJECT_DIR}/outputs/cache/target_context}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
EPOCHS="${EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-256}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"
MOLECULE_RATIO="${MOLECULE_RATIO:-4}"
TARGET_CONTEXT_RATIO="${TARGET_CONTEXT_RATIO:-1}"
MOLECULE_PHARMA_FP_CONDITION_PROBABILITY="${MOLECULE_PHARMA_FP_CONDITION_PROBABILITY:-0.5}"
TARGET_LIGAND_FRAGMENT_SELFIES_FOLDS="${TARGET_LIGAND_FRAGMENT_SELFIES_FOLDS:-10}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
USE_FSDP="${USE_FSDP:-false}"
FSDP_STRATEGY="${FSDP_STRATEGY:-full_shard}"

molexar_print_header "Molexar universal multi-condition SFT"
echo "Project: ${PROJECT_DIR}"
echo "Base model: ${BASE_MODEL_PATH}"
echo "Molecule context: ${MOLECULE_CONTEXT_PATH}"
echo "Output: ${OUTPUT_DIR}"

molexar_setup_environment
molexar_validate_paths \
    "${PROJECT_DIR}" \
    "${BASE_MODEL_PATH}" \
    "${MOLECULE_CONTEXT_PATH}" \
    "${PROPERTIES_PATH}" \
    "${PHARMA_FP_PATH}" \
    "${SAIR_INDEX_DIR}" \
    "${SAIR_STRUCTURES_DIR}" \
    "${SAIR_LIGAND_FRAGMENT_SELFIES_PATH}" \
    "${PLINDER_ROOT}" \
    "${PLINDER_INDEX_DIR}" \
    "${PLINDER_LIGAND_FRAGMENT_SELFIES_PATH}" \
    "${TOKENIZER_PATH}"
molexar_prepare_output "${OUTPUT_DIR}" "${TARGET_CONTEXT_CACHE_DIR}"
molexar_resolve_resume_args "${OUTPUT_DIR}" "${USE_FSDP}"
molexar_enter_project "${PROJECT_DIR}"

CMD=(
    accelerate launch
    --num_processes "${NUM_PROCESSES}"
    --mixed_precision "${MIXED_PRECISION}"
    scripts/run_training.py --task sft --sft_mode universal_multi
    --output_dir "${OUTPUT_DIR}"
    --base_model "${BASE_MODEL_PATH}"
    --molecule_context_file "${MOLECULE_CONTEXT_PATH}"
    --properties_file "${PROPERTIES_PATH}"
    --pharma_fp_file "${PHARMA_FP_PATH}"
    --sair_index_dir "${SAIR_INDEX_DIR}"
    --sair_structures_dir "${SAIR_STRUCTURES_DIR}"
    --sair_ligand_fragment_selfies_file "${SAIR_LIGAND_FRAGMENT_SELFIES_PATH}"
    --plinder_root "${PLINDER_ROOT}"
    --plinder_index_dir "${PLINDER_INDEX_DIR}"
    --plinder_ligand_fragment_selfies_file "${PLINDER_LIGAND_FRAGMENT_SELFIES_PATH}"
    --target_context_sources "${TARGET_CONTEXT_SOURCES}"
    --target_context_cache_dir "${TARGET_CONTEXT_CACHE_DIR}"
    --no_verify_target_pocket_paths
    --tokenizer_path "${TOKENIZER_PATH}"
    --molecule_ratio "${MOLECULE_RATIO}"
    --target_context_ratio "${TARGET_CONTEXT_RATIO}"
    --molecule_pharma_fp_condition_probability "${MOLECULE_PHARMA_FP_CONDITION_PROBABILITY}"
    --target_ligand_fragment_selfies_folds "${TARGET_LIGAND_FRAGMENT_SELFIES_FOLDS}"
    --batch_size "${BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --epochs "${EPOCHS}"
    --learning_rate "${LEARNING_RATE}"
    --warmup_steps "${WARMUP_STEPS}"
    --max_sequence_length "${MAX_SEQUENCE_LENGTH}"
    --dataloader_num_workers "${DATALOADER_WORKERS}"
)

if [[ "${USE_FSDP}" == "true" ]]; then
    CMD+=(--use_fsdp --fsdp_strategy "${FSDP_STRATEGY}")
fi
CMD+=("${RESUME_ARGS[@]}")

molexar_run "${CMD[@]}"
molexar_print_completion "${OUTPUT_DIR}"
