def train_stratified(
    input_csv: str,
    output_dir: str,
    output_name: str,
    num_samples: Union[int, float],
    stratify_by: str = "group_id",
    seed: Optional[int] = None
) -> None:
    pass


def merge_real_and_synthetic(
    real_csv: str,
    synth_csv: str,
    output_dir: str,
    output_name: str
) -> None:
    pass