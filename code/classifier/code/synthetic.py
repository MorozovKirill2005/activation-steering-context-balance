import numpy as np
import pandas as pd


NUM_CLASSES = 60
NUM_CONTEXTS = 6

METADATA_COLUMNS = [
    "path",
    "class_name",
    "class_id",
    "context_name",
    "context_id",
    "group_id",
]


def create_train_synth_csv(
    output_dir,
    metadata_prefix,
    synth_suffix,
    mode,
    seed=42,
):
    train = pd.read_csv(f"{output_dir}/{metadata_prefix}_train.csv")
    synth = pd.read_csv(f"{output_dir}/{metadata_prefix}_synth{synth_suffix}.csv")
    synth = synth[METADATA_COLUMNS].copy()

    train["is_synthetic"] = 0
    synth["is_synthetic"] = 1

    if mode == "balance_groups":
        target = len(train) // (NUM_CLASSES * NUM_CONTEXTS)
        rng = np.random.default_rng(seed)

        groups = []
        for _, group in train.groupby("group_id"):
            if len(group) > target:
                indices = rng.choice(
                    len(group),
                    size=target,
                    replace=False,
                )
                group = group.iloc[np.sort(indices)]

            groups.append(group)

        train = pd.concat(groups, ignore_index=True)

    train_synth = pd.concat([train, synth], ignore_index=True)
    train_synth.to_csv(
        f"{output_dir}/{metadata_prefix}_train_synth_{mode}{synth_suffix}.csv",
        index=False,
    )
