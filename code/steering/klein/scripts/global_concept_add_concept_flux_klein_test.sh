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
STEER_TXT="${STEER_TXT:-0}"
STRENGTH_TXT="${STRENGTH_TXT:-3}"
#!

USE_REFERENCE="${USE_REFERENCE:-0}" ##
IMAGE_SEED="${IMAGE_SEED:-42}" ##
METADATA_CSV="${METADATA_CSV:-../data/metadata_train.csv}" ##
IMAGE_ROOT="${IMAGE_ROOT:-..}" ##

MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.2-klein-4B}" ##
GS="${GS:-1.0}" ##
STEPS="${STEPS:-4}"
CONCEPT="${CONCEPT:-water}" #!
STRENGTH="${STRENGTH:-20}"
PROMPTS_PATH="${PROMPTS_PATH:-prompts_collection/add_global_concept/prompts.txt}"
VECTOR_DIR="${VECTOR_DIR:-experiments/$(basename "${MODEL_NAME}")/${CONCEPT// /_}/steering}" #!
## RESULTS_DIR="${RESULTS_DIR:-experiments/$(basename "${MODEL_NAME}")/${CONCEPT// /_}/add_s${STRENGTH}_${WIDTH}x${HEIGHT}_steps${STEPS}}" #!
## RESULTS_DIR="${RESULTS_DIR:-experiments/$(basename "${MODEL_NAME}")/${CONCEPT// /_}/add_s${STRENGTH}_qwen${STEER_TXT}_stxt${STRENGTH_TXT}_${WIDTH}x${HEIGHT}_steps${STEPS}}"
##
if [[ "${USE_REFERENCE}" != "0" && "${USE_REFERENCE}" != "1" ]]; then
    echo "USE_REFERENCE must be 0 or 1." >&2
    exit 1
fi

MODE="ref0"

if [[ "${USE_REFERENCE}" == "1" ]]; then
    MODE="ref1_pick${IMAGE_SEED}"
fi

PROMPT_TAG="$(basename "${PROMPTS_PATH}")"
PROMPT_TAG="${PROMPT_TAG%.*}"

RESULTS_DIR="${RESULTS_DIR:-experiments/$(basename "${MODEL_NAME}")/${CONCEPT// /_}/native_reference/${PROMPT_TAG}_${PROMPT_FORMAT}/add_s${STRENGTH}_qwen${STEER_TXT}_stxt${STRENGTH_TXT}_${WIDTH}x${HEIGHT}_gs${GS}_steps${STEPS}/${MODE}_seed${SEED}_n${NUM_PROMPTS}x${SAMPLES_PER_PROMPT}}"
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