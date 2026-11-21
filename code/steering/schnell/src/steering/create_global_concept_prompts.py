from pathlib import Path


def read_prompts(prompt_path="prompts.txt"):
    with Path(prompt_path).open(encoding="utf-8-sig") as file:
        classes = [line.strip() for line in file if line.strip()]
    if not classes:
        raise ValueError(f"No class names found in {prompt_path}")
    # if len(classes) != len(set(classes)):
    #     raise ValueError(f"Duplicate class names in {prompt_path}")
    return classes


def get_global_concept_prompts(
    num=60,
    concept_pos="water",
    concept_neg=None,
    prompt_path="prompts.txt",
):
    if concept_pos not in ("water", "grass", "rock", "autumn", "outdoor", "dim") or concept_neg not in (None, ""):
        raise ValueError(
            "No valide concept_pos or concept_neg"
        )
    classes = read_prompts(prompt_path)
    if num is None:
        num = len(classes)
    if not 1 <= num <= len(classes):
        raise ValueError(
            f"Requested {num} classes, "
            f"but {prompt_path} contains {len(classes)}."
        )
    if concept_pos in ("water", "grass", "autumn"):
        suffix = f"with {concept_pos} in the background."
    elif concept_pos == "rock":
        suffix = f"with rocks in the background."
    elif concept_pos == "outdoor":
        suffix = "outdoor."
    elif concept_pos == "dim":
        suffix = "in dim lighting."
    neutral = []
    positive = []
    for name in classes[:num]:
        name = " ".join(name.replace("_", " ").split())
        if not name:
            raise ValueError(
                "A class name becomes empty after replacing underscores."
            )
        article = "an" if name[0].lower() in "aeiou" else "a"
        base = f"A realistic photo of {article} {name}"
        neutral.append(f"{base}.")
        positive.append(f"{base}, {suffix}")
    return positive, neutral