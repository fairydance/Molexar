#!/usr/bin/env bash
set -euo pipefail

molexar_print_command() {
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n'
}

molexar_run() {
    molexar_print_command "$@"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY_RUN=1, command not executed."
        return 0
    fi
    "$@"
}

molexar_setup_environment() {
    if [[ -n "${ENV_SETUP_SCRIPT:-}" ]]; then
        source "${ENV_SETUP_SCRIPT}"
    fi
    if [[ -n "${CONDA_ACTIVATE_SCRIPT:-}" ]]; then
        source "${CONDA_ACTIVATE_SCRIPT}" "${CONDA_ENV:-molexar}"
    fi
}

molexar_validate_paths() {
    if [[ "${DRY_RUN:-0}" == "1" || "${VALIDATE_PATHS:-1}" != "1" ]]; then
        return 0
    fi
    local missing=0
    for path in "$@"; do
        if [[ ! -e "${path}" ]]; then
            echo "Missing required path: ${path}" >&2
            missing=1
        fi
    done
    if [[ "${missing}" == "1" ]]; then
        exit 1
    fi
}

molexar_prepare_output() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        return 0
    fi
    mkdir -p "$@"
}

molexar_enter_project() {
    local project_dir="$1"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        return 0
    fi
    cd "${project_dir}"
}

molexar_resolve_resume_args() {
    local output_dir="$1"
    local use_fsdp="$2"
    RESUME_ARGS=()

    if [[ "${use_fsdp}" != "true" ]]; then
        RESUME_ARGS=(--no_resume)
        return 0
    fi

    shopt -s nullglob
    local checkpoints=("${output_dir}"/checkpoints/checkpoint-*)
    shopt -u nullglob

    if (( ${#checkpoints[@]} == 0 )); then
        RESUME_ARGS=(--no_resume)
    fi
}

molexar_resolve_max_steps_args() {
    local max_steps="$1"
    MAX_STEPS_ARGS=()
    if [[ "${max_steps}" != "-1" ]]; then
        MAX_STEPS_ARGS=(--max_steps "${max_steps}")
    fi
}

molexar_print_header() {
    local title="$1"
    echo "${title}"
}

molexar_print_completion() {
    local output_dir="$1"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "Dry run complete. Output would be: ${output_dir}"
    else
        echo "Training complete. Output: ${output_dir}"
    fi
}
