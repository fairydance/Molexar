#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MODEL_NAME="${MODEL_NAME:-10m_256h_16l}"

DATA_PATH="${DATA_PATH:-/path/to/pretrain.fragment_selfies}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/models/configs/base/config_${MODEL_NAME}.json}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${PROJECT_DIR}/models/tokenizer}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/pretrain_base_${MODEL_NAME}}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
EPOCHS="${EPOCHS:-2}"
MAX_STEPS="${MAX_STEPS:--1}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-256}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
USE_FSDP="${USE_FSDP:-false}"
FSDP_STRATEGY="${FSDP_STRATEGY:-full_shard}"

molexar_print_header "Molexar base pretraining"
echo "Project: ${PROJECT_DIR}"
echo "Data: ${DATA_PATH}"
echo "Config: ${CONFIG_PATH}"
echo "Output: ${OUTPUT_DIR}"

molexar_setup_environment
molexar_validate_paths "${PROJECT_DIR}" "${DATA_PATH}" "${CONFIG_PATH}" "${TOKENIZER_PATH}"
molexar_prepare_output "${OUTPUT_DIR}"
molexar_resolve_resume_args "${OUTPUT_DIR}" "${USE_FSDP}"
molexar_resolve_max_steps_args "${MAX_STEPS}"
molexar_enter_project "${PROJECT_DIR}"

CMD=(
    accelerate launch
    --num_processes "${NUM_PROCESSES}"
    --mixed_precision "${MIXED_PRECISION}"
    scripts/run_training.py --task pretrain
    --output_dir "${OUTPUT_DIR}"
    --config_path "${CONFIG_PATH}"
    --train_data_path "${DATA_PATH}"
    --tokenizer_path "${TOKENIZER_PATH}"
    --batch_size "${BATCH_SIZE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --epochs "${EPOCHS}"
    --learning_rate "${LEARNING_RATE}"
    --warmup_steps "${WARMUP_STEPS}"
    --max_sequence_length "${MAX_SEQUENCE_LENGTH}"
    --dataloader_num_workers "${DATALOADER_WORKERS}"
    "${MAX_STEPS_ARGS[@]}"
)

if [[ "${USE_FSDP}" == "true" ]]; then
    CMD+=(--use_fsdp --fsdp_strategy "${FSDP_STRATEGY}")
fi
CMD+=("${RESUME_ARGS[@]}")

molexar_run "${CMD[@]}"
molexar_print_completion "${OUTPUT_DIR}"
