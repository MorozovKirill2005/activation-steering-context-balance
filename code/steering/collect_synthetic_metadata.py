import argparse
import json
import re
from pathlib import Path

import pandas as pd


CONTEXT_NAMES = ["autumn", "dim", "grass", "outdoor", "rock", "water"]
RESULTS_PATHS = {
    "schnell": "FLUX.1-schnell/{context}/add_s20_512x512_steps4/steered",
    "klein": (
        "FLUX.2-klein-4B/{context}/native_reference/full_text/"
        "add_s80_qwen1_stxt1_512x512_gs1.0_steps4/ref1_pick42/steered"
    ),
}


def normalize_name(name):
    return " ".join(name.replace("_", " ").lower().split())


def get_class_name(prompt, class_names):
    prompt = prompt.replace("_", " ").strip()
    prompt = re.sub(
        r"^(?:Create a new photograph|A realistic photo|A photo) of (?:a|an) ",
        "",
        prompt,
    )
    name = prompt.split(".", 1)[0].split(",", 1)[0].strip()
    return class_names[normalize_name(name)]


def main():
    steering_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["schnell", "klein"])
    parser.add_argument("--output_dir", type=Path, default=steering_dir.parent / "data")
    parser.add_argument("--metadata_prefix", default="metadata")
    parser.add_argument("--class_mapping", type=Path, default=None)
    parser.add_argument("--results_template", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    mapping_path = args.class_mapping or steering_dir.parent / "data/nicopp/dg_label_id_mapping.json"
    with mapping_path.open(encoding="utf-8-sig") as file:
        class_to_id = json.load(file)
    class_names = {normalize_name(name): name for name in class_to_id}
    template = args.results_template or RESULTS_PATHS[args.mode]
    experiments_dir = steering_dir / args.mode / "experiments"

    rows = []
    for context_id, context_name in enumerate(CONTEXT_NAMES):
        folder = experiments_dir / template.format(context=context_name)
        count = 0
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() != ".png":
                continue

            index, name = path.stem.split("_", 1)
            prompt, name = name.split("_seed_", 1)
            seed = int(name.split("_", 1)[0])

            if args.mode == "klein":
                with path.with_suffix(".json").open(encoding="utf-8-sig") as file:
                    prompt = json.load(file)["prompt"]

            class_name = get_class_name(prompt, class_names)
            class_id = int(class_to_id[class_name])
            rows.append({
                "path": str(path.resolve()),
                "class_name": class_name,
                "class_id": class_id,
                "context_name": context_name,
                "context_id": context_id,
                "group_id": class_id * len(CONTEXT_NAMES) + context_id,
                "is_synthetic": 1,
                "seed": seed,
                "index": int(index),
            })
            count += 1
        print(f"{context_name}: {count}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.metadata_prefix}_synth_{args.mode}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {len(rows)} images to {output_path}")


if __name__ == "__main__":
    main()
