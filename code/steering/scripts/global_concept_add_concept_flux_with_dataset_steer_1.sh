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
NUM_PROMPTS="${NUM_PROMPTS:-60}"
PROMPT_FORMAT="${PROMPT_FORMAT:-text}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
SEED="${SEED:-42}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
STEER_TXT="${STEER_TXT:-0}"
STRENGTH_TXT="${STRENGTH_TXT:-3}"

IMAGE_INIT="${IMAGE_INIT:-1}"
LATENT_STEPS="${LATENT_STEPS:-4}"
IMAGE_SEED="${IMAGE_SEED:-42}"
METADATA_CSV="${METADATA_CSV:-../data/metadata_train.csv}"
IMAGE_ROOT="${IMAGE_ROOT:-..}"
#!

MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.1-schnell}"
GS="${GS:-0.0}"
STEPS="${STEPS:-4}"
CONCEPT="${CONCEPT:-water}" #!
STRENGTH="${STRENGTH:-20}"
PROMPTS_PATH="${PROMPTS_PATH:-prompts_collection/add_global_concept/prompts.txt}" #!
VECTOR_DIR="${VECTOR_DIR:-experiments/$(basename "${MODEL_NAME}")/${CONCEPT// /_}/steering}" #!

MODE="txt2img"
if [[ "$IMAGE_INIT" == "1" ]]; then
    MODE="latent${LATENT_STEPS}_pick${IMAGE_SEED}"
fi
RESULTS_DIR="${RESULTS_DIR:-experiments/$(basename "${MODEL_NAME}")/${CONCEPT// /_}/add_s${STRENGTH}_${WIDTH}x${HEIGHT}_steps${STEPS}/${MODE}}"

#!
TXT_ARGS=()

if [[ "${STEER_TXT}" == "1" && "${CONCEPT}" != "baseline" ]]; then
    TXT_ARGS+=(--steer_txt --strength_txt "${STRENGTH_TXT}")
fi
#!

#!
if [[ "$IMAGE_INIT" != "0" && "$IMAGE_INIT" != "1" ]]; then
    echo "IMAGE_INIT must be 0 or 1." >&2
    exit 1
fi
DATASET_ARGS=()
if [[ "$IMAGE_INIT" == "1" ]]; then
    if [[ ! "$LATENT_STEPS" =~ ^[1-9][0-9]*$ ]] \
        || (( LATENT_STEPS > STEPS )); then
        echo "LATENT_STEPS must be between 1 and $STEPS." >&2
        exit 1
    fi
    DATASET_ARGS+=(
        --image_init
        --latent_steps "$LATENT_STEPS"
        --image_seed "$IMAGE_SEED"
    )
    if (( LATENT_STEPS < STEPS )); then
        if [[ ! -f "$METADATA_CSV" ]]; then
            echo "Set METADATA_CSV to metadata_train.csv." >&2
            exit 1
        fi
        DATASET_ARGS+=(--metadata_csv "$METADATA_CSV")
        if [[ -n "$IMAGE_ROOT" ]]; then
            DATASET_ARGS+=(--image_root "$IMAGE_ROOT")
        fi
    fi
fi
#!

python ./src/steering/apply_steering_with_injection_flux_with_dataset_steer_1.py \
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
    "${DATASET_ARGS[@]}"