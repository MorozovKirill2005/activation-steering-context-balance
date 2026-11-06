import json
import os

import matplotlib.pyplot as plt
import numpy as np

from .data import NUM_CLASSES, NUM_CONTEXTS, CONTEXT_NAMES


def get_group_accuracy_heatmap(
    group_accuracy: np.ndarray,
    data_dir: str,
):
    if group_accuracy.shape != (NUM_CONTEXTS, NUM_CLASSES):
        raise ValueError(
            f"Expected group_accuracy shape "
            f"({NUM_CONTEXTS}, {NUM_CLASSES}), "
            f"got {group_accuracy.shape}"
        )

    mapping_path = os.path.join(
        data_dir,
        "dg_label_id_mapping.json",
    )

    with open(mapping_path, "r", encoding="utf-8") as file:
        class_to_id = json.load(file)

    class_names = [str(i) for i in range(NUM_CLASSES)]

    for class_name, class_id in class_to_id.items():
        class_names[int(class_id)] = class_name

    masked_accuracy = np.ma.masked_invalid(
        group_accuracy
    )

    fig, ax = plt.subplots(
        figsize=(24, 6),
        dpi=120,
    )

    heatmap = ax.imshow(
        masked_accuracy,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    ax.set_xlabel("Class")
    ax.set_ylabel("Context")
    ax.set_title("Group Accuracy")

    ax.set_xticks(
        np.arange(NUM_CLASSES)
    )
    ax.set_xticklabels(
        class_names,
        rotation=90,
        ha="center",
        fontsize=7,
    )

    ax.set_yticks(
        np.arange(NUM_CONTEXTS)
    )
    ax.set_yticklabels(
        CONTEXT_NAMES,
        rotation=0,
    )

    colorbar = fig.colorbar(
        heatmap,
        ax=ax,
    )
    colorbar.set_label("Accuracy")

    for context_id in range(NUM_CONTEXTS):
        for class_id in range(NUM_CLASSES):
            value = group_accuracy[
                context_id,
                class_id
            ]

            if np.isnan(value):
                continue

            ax.text(
                class_id,
                context_id,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=5,
                color=(
                    "white"
                    if value < 0.25 or value > 0.8
                    else "black"
                ),
            )

    fig.tight_layout()

    return fig


def get_class_accuracy_bar(
    class_accuracy: np.ndarray,
    data_dir: str,
):
    if class_accuracy.shape != (NUM_CLASSES,):
        raise ValueError(
            f"Expected class_accuracy shape "
            f"({NUM_CLASSES},), "
            f"got {class_accuracy.shape}"
        )

    mapping_path = os.path.join(
        data_dir,
        "dg_label_id_mapping.json",
    )

    with open(mapping_path, "r", encoding="utf-8") as file:
        class_to_id = json.load(file)

    class_names = [str(i) for i in range(NUM_CLASSES)]

    for class_name, class_id in class_to_id.items():
        class_names[int(class_id)] = class_name

    class_ids = np.arange(NUM_CLASSES)

    fig, ax = plt.subplots(
        figsize=(24, 7),
        dpi=120,
    )

    ax.bar(
        class_ids,
        class_accuracy,
    )

    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_title("Class Accuracy")

    # Именно фиксированный диапазон,
    # чтобы графики разных эпох были сравнимы.
    ax.set_ylim(0.0, 1.0)

    ax.set_xticks(class_ids)
    ax.set_xticklabels(
        class_names,
        rotation=90,
        ha="center",
        fontsize=7,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    return fig


def get_dataset_distribution_heatmap(
    dataset,
):
    distribution = np.zeros(
        (NUM_CONTEXTS, NUM_CLASSES),
        dtype=np.int64,
    )

    np.add.at(
        distribution,
        (
            dataset.context_ids,
            dataset.class_ids,
        ),
        1,
    )

    mapping_path = os.path.join(
        dataset.data_dir,
        "dg_label_id_mapping.json",
    )

    with open(mapping_path, "r", encoding="utf-8") as file:
        class_to_id = json.load(file)

    class_names = [str(i) for i in range(NUM_CLASSES)]

    for class_name, class_id in class_to_id.items():
        class_names[int(class_id)] = class_name

    max_count = int(distribution.max())

    fig, ax = plt.subplots(
        figsize=(24, 6),
        dpi=120,
    )

    heatmap = ax.imshow(
        distribution,
        cmap="viridis",
        vmin=0,
        vmax=max(max_count, 1),
        aspect="auto",
    )

    ax.set_xlabel("Class")
    ax.set_ylabel("Context")
    ax.set_title("Dataset Distribution")

    ax.set_xticks(
        np.arange(NUM_CLASSES)
    )
    ax.set_xticklabels(
        class_names,
        rotation=90,
        ha="center",
        fontsize=7,
    )

    ax.set_yticks(
        np.arange(NUM_CONTEXTS)
    )
    ax.set_yticklabels(
        CONTEXT_NAMES,
        rotation=0,
    )

    colorbar = fig.colorbar(
        heatmap,
        ax=ax,
    )
    colorbar.set_label("Number of Samples")

    for context_id in range(NUM_CONTEXTS):
        for class_id in range(NUM_CLASSES):
            value = distribution[
                context_id,
                class_id
            ]

            ax.text(
                class_id,
                context_id,
                str(value),
                ha="center",
                va="center",
                fontsize=5,
                color=(
                    "white"
                    if value > max_count * 0.55
                    else "black"
                ),
            )

    fig.tight_layout()

    return fig
