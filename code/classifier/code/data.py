import os
from typing import Callable, Optional
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import json
import gdown
import random
from pathlib import Path
from zipfile import ZipFile


NUM_CLASSES = 60
NUM_CONTEXTS = 6
CONTEXT_NAMES = ['autumn', 'dim', 'grass', 'outdoor', 'rock', 'water']


TRANSFORMS = {
    "base": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
}


def generate_nicopp_split(
    data_dir: str,
    output_dir: str,
    metadata_prefix: str = "metadata",
    download: bool = True,
    seed: int = 42,
    num_samples_val_test: int = 75,
    verbose: bool = True
) -> None:
    
    if download:
        os.makedirs(data_dir, exist_ok=True)

        zip_path = os.path.join(data_dir, "track_1.zip")
        gdown.download(
            "https://www.dropbox.com/sh/u2bq2xo8sbax4pr/AACvgYpfQHfS7u_M3yJwkK-ra/track_1/track_1.zip?dl=1",
            zip_path,
            quiet=False
        )
        zf = ZipFile(zip_path, "r")
        zf.extractall(data_dir)
        zf.close()
        os.remove(zip_path)

        json_path = os.path.join(data_dir, "dg_label_id_mapping.json")
        gdown.download(
            "https://www.dropbox.com/sh/u2bq2xo8sbax4pr/AADMZPkoNJVI4IP1qbLuKtlFa/track_1/dg_label_id_mapping.json?dl=1",
            json_path,
            quiet=False
        )

    data_dir = Path(data_dir)
    class_metadata = json.load(open(data_dir/'dg_label_id_mapping.json', 'r'))
    
    all_data = []
    for context_id, context_name in enumerate(CONTEXT_NAMES):
        for class_name in class_metadata:
            folder_path = data_dir/'public_dg_0416'/'train'/context_name/class_name
            class_id = class_metadata[class_name]
            group_id = class_id * NUM_CONTEXTS + context_id
            for img_path in Path(folder_path).glob('*.jpg'):
                all_data.append({
                    'path': img_path,
                    'class_name': class_name,
                    'class_id': class_id,
                    'context_name': context_name,
                    'context_id': context_id,
                    'group_id': group_id
                })
    df = pd.DataFrame(all_data)

    rng = random.Random(seed)
    val_set, test_set = [], []
    for group_id in pd.unique(df['group_id']):
        df_group = df[df['group_id'] == group_id]
        curr_data = df_group['path'].values
        rng.shuffle(curr_data)
        split_size = min(len(curr_data), num_samples_val_test)
        val_set.extend(curr_data[:split_size // 3])
        test_set.extend(curr_data[split_size // 3:split_size])
    if verbose:
        print(f"Val: {len(val_set)}\nTest: {len(test_set)}")
    assert len(set(val_set).intersection(set(test_set))) == 0

    val_set = set(val_set)
    test_set = set(test_set)
    train_mask = ~df["path"].isin(val_set | test_set)
    val_mask = df["path"].isin(val_set)
    test_mask = df["path"].isin(test_set)

    os.makedirs(output_dir, exist_ok=True)

    df.loc[train_mask].to_csv(os.path.join(output_dir, f"{metadata_prefix}_train.csv"), index=False)
    df.loc[val_mask].to_csv(os.path.join(output_dir, f"{metadata_prefix}_val.csv"), index=False)
    df.loc[test_mask].to_csv(os.path.join(output_dir, f"{metadata_prefix}_test.csv"), index=False)


class NICOPPDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        data_dir: str,
        transform: Optional[Callable] = None,
        has_synthetic: bool = False,
        num_contexts: int = NUM_CONTEXTS
    ):
        super().__init__()
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.transform = transform
        self.has_synthetic = has_synthetic
        self.num_contexts = num_contexts

        df = pd.read_csv(csv_path)
        required = {"path", "class_name", "context_name", "class_id", "context_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV {csv_path} не содержит колонок: {missing}")

        self.paths = df["path"].tolist()
        self.class_ids = df["class_id"].to_numpy(dtype="int64")
        self.context_ids = df["context_id"].to_numpy(dtype="int64")
        self.group_ids = df["group_id"].to_numpy(dtype="int64")
        if self.has_synthetic:
            self.sample_is_synth = df["is_synthetic"].to_numpy(dtype="int64")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        class_id = int(self.class_ids[idx])
        context_id = int(self.context_ids[idx])
        group_id = int(self.group_ids[idx])
        if self.has_synthetic:
            is_synth = int(self.sample_is_synth[idx])
        else:
            is_synth = 0

        return {
            "image": image,
            "class_id": class_id,
            "context_id": context_id,
            "group_id": group_id,
            "is_synthetic": is_synth
        }


class NICOPPDataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        balance_synthetic: bool = False,
        num_workers: int = 0
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.balance_synthetic = balance_synthetic
        self.num_workers = num_workers

        if self.balance_synthetic:
            pass

        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)


def data_load(config):
    if config["gen_split"]:
        generate_nicopp_split(
            data_dir=config["data_dir"],
            output_dir=config["output_dir"],
            metadata_prefix=config["metadata_prefix"],
            download=config["download"],
            seed=config["split_seed"]
        )

    train_dataset = NICOPPDataset(
        csv_path=f'{config["output_dir"]}/{config["metadata_prefix"]}_train.csv',
        data_dir=config["data_dir"],
        transform=TRANSFORMS[config["transform"]["train"]],
        has_synthetic=config["has_synthetic"]
    )
    val_dataset = NICOPPDataset(
        csv_path=f'{config["output_dir"]}/{config["metadata_prefix"]}_val.csv',
        data_dir=config["data_dir"],
        transform=TRANSFORMS[config["transform"]["val"]],
        has_synthetic=config["has_synthetic"]
    )
    test_dataset = NICOPPDataset(
        csv_path=f'{config["output_dir"]}/{config["metadata_prefix"]}_test.csv',
        data_dir=config["data_dir"],
        transform=TRANSFORMS[config["transform"]["test"]],
        has_synthetic=config["has_synthetic"]
    )

    train_loader = NICOPPDataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"]["train"],
        shuffle=True,
        balance_synthetic=config["balance_synthetic"],
        num_workers=config["num_workers"]
    )
    val_loader = NICOPPDataLoader(
        dataset=val_dataset,
        batch_size=config["batch_size"]["val"],
        shuffle=False,
        balance_synthetic=False,
        num_workers=config["num_workers"]
    )
    test_loader = NICOPPDataLoader(
        dataset=test_dataset,
        batch_size=config["batch_size"]["test"],
        shuffle=False,
        balance_synthetic=False,
        num_workers=config["num_workers"]
    )

    return (train_loader, val_loader, test_loader)
