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

MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.1-schnell}"
GS="${GS:-0.0}"
STEPS="${STEPS:-4}"
CONCEPT="${CONCEPT:-water}" #!
STRENGTH="${STRENGTH:-20}"
#! PROMPTS_PATH="${PROMPTS_PATH:-prompts_collection/add_global_concept/prompts.txt}" #!
VECTOR_DIR="${VECTOR_DIR:-experiments/$(basename "${MODEL_NAME}")/${CONCEPT// /_}/steering}" #!
RESULTS_DIR="${RESULTS_DIR:-experiments/$(basename "${MODEL_NAME}")/${CONCEPT// /_}/add_s${STRENGTH}_${WIDTH}x${HEIGHT}_steps${STEPS}}" #!

#!
SEED_ARGS=()

if [[ "${CONCEPT}" != "baseline" ]]; then
    #! PROMPT_FORMAT=classes
    PROMPTS_PATH="${PROMPTS_PATH:-prompts_collection/add_global_concept/${CONCEPT}_prompts.txt}"
    SEEDS_PATH="${SEEDS_PATH:-prompts_collection/add_global_concept/${CONCEPT}_seeds.txt}"

    NUM_PROMPTS="$(python - "${PROMPTS_PATH}" "${SEEDS_PATH}" <<'PY'
import sys
from pathlib import Path

def read_lines(path):
    with Path(path).open(encoding="utf-8-sig") as file:
        return [line.strip() for line in file if line.strip()]

prompts = read_lines(sys.argv[1])
seeds = [int(value) for value in read_lines(sys.argv[2])]

if len(prompts) != len(seeds):
    raise ValueError("Prompt and seed counts must match")

if any(seed < 0 or seed >= 2**64 for seed in seeds):
    raise ValueError("Seeds must be in [0, 2**64 - 1]")

print(len(prompts))
PY
)"

    if (( NUM_PROMPTS == 0 )); then
        echo "No images to generate for ${CONCEPT}."
        exit 0
    fi

    SEED_ARGS+=(--seeds_path "${SEEDS_PATH}")

    echo "Prompts: ${PROMPTS_PATH}; seeds: ${SEEDS_PATH}; jobs: ${NUM_PROMPTS}"
else
    PROMPTS_PATH="${PROMPTS_PATH:-prompts_collection/add_global_concept/prompts.txt}"
    NUM_PROMPTS="${NUM_PROMPTS:-60}"

    if [[ -n "${SEEDS_PATH:-}" ]]; then
        SEED_ARGS+=(--seeds_path "${SEEDS_PATH}")
    fi
fi
#!

#!
TXT_ARGS=()

if [[ "${STEER_TXT}" == "1" && "${CONCEPT}" != "baseline" ]]; then
    TXT_ARGS+=(--steer_txt --strength_txt "${STRENGTH_TXT}")
fi
#!

python ./src/steering/apply_steering_with_injection_flux.py \
    --model_name "${MODEL_NAME}" \
    --data_dir "${VECTOR_DIR}" \
    --task "add concept" \
    --prompts_path "${PROMPTS_PATH}" \
    "${SEED_ARGS[@]}" \
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
    "${TXT_ARGS[@]}"