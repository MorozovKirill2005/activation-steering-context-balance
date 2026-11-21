#!/bin/bash
# Step 3/3 (FLUX): concept addition / switching (e.g. add a smile, woman -> man).
# The same mean-difference field is applied with a positive sign; no classifier.
# Build the vector first, e.g. for "smile":
#   CONCEPT=smile TASK=people NUM_PROMPTS=20 bash scripts/get_vector.sh
#   CONCEPT=smile NUM_PROMPTS=20 bash scripts/steering_calculate.sh
#
#   CONCEPT=smile STRENGTH=20 bash scripts/add_concept_flux.sh
set -euo pipefail


#!
#! NUM_PROMPTS="${NUM_PROMPTS:-60}"
PROMPT_FORMAT="${PROMPT_FORMAT:-text}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
SEED="${SEED:-42}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
STEER_TXT="${STEER_TXT:-1}"
STRENGTH_TXT="${STRENGTH_TXT:-1}"
#!

USE_REFERENCE="${USE_REFERENCE:-1}" ##
IMAGE_SEED="${IMAGE_SEED:-42}" ##
METADATA_CSV="${METADATA_CSV:-../data/metadata_train.csv}" ##
IMAGE_ROOT="${IMAGE_ROOT:-..}" ##

MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.2-klein-4B}" ##
GS="${GS:-1.0}" ##
STEPS="${STEPS:-4}"
CONCEPT="${CONCEPT:-water}" #!
STRENGTH="${STRENGTH:-80}"

##
VERSION="${VERSION:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-800}"
if [[ ! "${VERSION}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "VERSION must be a nonnegative integer without leading zeros" >&2
    exit 1
fi
if [[ ! "${CHUNK_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CHUNK_SIZE must be a positive integer" >&2
    exit 1
fi
START_INDEX=$((VERSION * CHUNK_SIZE))
PROMPTS_PATH="${PROMPTS_PATH:-prompts_collection/add_global_concept/${CONCEPT}_prompt_with_reference_style_${VERSION}.txt}"
SEEDS_PATH="${SEEDS_PATH:-prompts_collection/add_global_concept/${CONCEPT}_seeds_with_reference_style_${VERSION}.txt}"
##

VECTOR_DIR="${VECTOR_DIR:-experiments/$(basename "${MODEL_NAME}")/${CONCEPT// /_}/steering}" #!
## RESULTS_DIR="${RESULTS_DIR:-experiments/$(basename "${MODEL_NAME}")/${CONCEPT// /_}/add_s${STRENGTH}_${WIDTH}x${HEIGHT}_steps${STEPS}}" #!
## RESULTS_DIR="${RESULTS_DIR:-experiments/$(basename "${MODEL_NAME}")/${CONCEPT// /_}/add_s${STRENGTH}_qwen${STEER_TXT}_stxt${STRENGTH_TXT}_${WIDTH}x${HEIGHT}_steps${STEPS}}"
##
if [[ "${USE_REFERENCE}" != "0" && "${USE_REFERENCE}" != "1" ]]; then
    echo "USE_REFERENCE must be 0 or 1." >&2
    exit 1
fi

MODE="ref0"

for input_file in "${PROMPTS_PATH}" "${SEEDS_PATH}"; do
    if [[ ! -f "${input_file}" || ! -r "${input_file}" || ! -s "${input_file}" ]]; then
        echo "Input file is missing, unreadable or empty: ${input_file}" >&2
        exit 1
    fi
done

TOTAL_PROMPTS="$(awk 'NF {n++} END {print n+0}' "${PROMPTS_PATH}")"
TOTAL_SEEDS="$(awk 'NF {n++} END {print n+0}' "${SEEDS_PATH}")"

if [[ "${TOTAL_PROMPTS}" -eq 0 || "${TOTAL_PROMPTS}" -ne "${TOTAL_SEEDS}" ]]; then
    echo "Prompt and seed counts must match and be nonzero" >&2
    exit 1
fi

NUM_PROMPTS="${NUM_PROMPTS:-${TOTAL_PROMPTS}}"

if [[ ! "${NUM_PROMPTS}" =~ ^[1-9][0-9]*$ ]] \
    || [ "${NUM_PROMPTS}" -gt "${TOTAL_PROMPTS}" ]; then
    echo "NUM_PROMPTS must be between 1 and ${TOTAL_PROMPTS}" >&2
    exit 1
fi

echo "Context: ${CONCEPT}; version: ${VERSION}; start_index: ${START_INDEX}; generating ${NUM_PROMPTS}/${TOTAL_PROMPTS} prompts"

if [[ "${USE_REFERENCE}" == "1" ]]; then
    MODE="ref1_pick${IMAGE_SEED}"
fi

PROMPT_TAG="$(basename "${PROMPTS_PATH}")"
PROMPT_TAG="${PROMPT_TAG%.*}"

RESULTS_DIR="${RESULTS_DIR:-experiments/$(basename "${MODEL_NAME}")/${CONCEPT// /_}/native_reference/full_${PROMPT_FORMAT}/add_s${STRENGTH}_qwen${STEER_TXT}_stxt${STRENGTH_TXT}_${WIDTH}x${HEIGHT}_gs${GS}_steps${STEPS}/${MODE}}"
##

#!
TXT_ARGS=()

if [[ "${STEER_TXT}" == "1" && "${CONCEPT}" != "baseline" ]]; then
    TXT_ARGS+=(--steer_txt --strength_txt "${STRENGTH_TXT}")
fi
#!

##
REFERENCE_ARGS=()

if [[ "${USE_REFERENCE}" == "1" ]]; then
    REFERENCE_ARGS+=(
        --use_reference
        --metadata_csv "${METADATA_CSV}"
        --image_root "${IMAGE_ROOT}"
        --image_seed "${IMAGE_SEED}"
    )
fi
##

python ./src/steering/apply_steering_with_injection_flux_klein.py \
    --model_name "${MODEL_NAME}" \
    --data_dir "${VECTOR_DIR}" \
    --task "add concept" \
    --prompts_path "${PROMPTS_PATH}" \
    --seeds_path "${SEEDS_PATH}" \
    --start_index "${START_INDEX}" \
    --injection_point block \
    --vector_type diff \
    --steering_type separate \
    --strength "${STRENGTH}" \
    --strength_img 0.0 \
    --inference_steps "${STEPS}" \
    --guidance_scale "${GS}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --results_dir "${RESULTS_DIR}" \
    --num_prompts "${NUM_PROMPTS}" \
    --concept "${CONCEPT}" \
    --prompt_format "${PROMPT_FORMAT}" \
    --seed "${SEED}" \
    --num_samples_per_prompt "${SAMPLES_PER_PROMPT}" \
    "${TXT_ARGS[@]}" \
    "${REFERENCE_ARGS[@]}"