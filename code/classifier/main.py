import argparse
import json
import os

import torch
import torch.nn as nn
from code.utils import set_global_seed, config_transform
from code.data import data_load
from code.models import ResNet50
from code.train import train_val, optimizer_load


os.environ["COMET_API_KEY"] = ""


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def experiment(config):
    set_global_seed(config["seed"])

    data_loader = data_load(config["data"])

    net = ResNet50(**config["model"]).to(config["device"])

    optimizer = optimizer_load(config["optimizer"], net)

    train_val(
        network=net,
        n_epochs=config["n_epochs"],
        criterion=nn.CrossEntropyLoss(reduction='none'),
        optimizer=optimizer,
        data_loader=data_loader,
        model_selection=config["model_selection"],
        logging_interval=config["logging_interval"],
        log_dir=config["log_dir"],
        config=config_transform(config),
        device=config["device"],
        verbose=config["verbose"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    config.setdefault("device", DEVICE)
    experiment(config)
