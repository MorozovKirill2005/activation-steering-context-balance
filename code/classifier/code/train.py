import os
from typing import Optional
import torch
import comet_ml
import matplotlib.pyplot as plt
import numpy as np

from .visualization import get_group_accuracy_heatmap, get_class_accuracy_bar, get_dataset_distribution_heatmap
from .metrics import metrics, get_selection_value, is_better_metric


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


OPTIMIZERS = {
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW
}


def optimizer_load(config, net):
    return OPTIMIZERS[config["name"]](net.parameters(), **config["parameters"])


def log_dataset_distribution(
    dataset,
    logger,
    split: str
):
    figure = get_dataset_distribution_heatmap(dataset)
    logger.log_figure(
        figure_name=f"{split}/distribution",
        figure=figure
    )
    plt.close(figure)


def log_val_metrics(
    split: str,
    epoch: int,
    val_metrics: dict,
    group_accuracy: np.ndarray,
    class_accuracy: np.ndarray,
    data_dir: str,
    logger
):
    for metric_name, metric_value in val_metrics.items():
        logger.log_metric(
            f"{split}/{metric_name}",
            metric_value,
            step=epoch,
        )

    group_accuracy_figure = get_group_accuracy_heatmap(
        group_accuracy=group_accuracy,
        data_dir=data_dir
    )
    logger.log_figure(
        f"{split}/group_accuracy",
        figure=group_accuracy_figure,
        step=epoch
    )
    plt.close(group_accuracy_figure)

    class_accuracy_figure = get_class_accuracy_bar(
        class_accuracy=class_accuracy,
        data_dir=data_dir
    )
    logger.log_figure(
        f"{split}/class_accuracy",
        figure=class_accuracy_figure,
        step=epoch
    )
    plt.close(class_accuracy_figure)


def train_epoch(
    network,
    train_loader,
    criterion,
    optimizer,
    device=DEVICE,
    epoch: Optional[int] = None,
    logging_interval: Optional[int] = None,
    logger=None
):
    network.train()

    total_loss = 0.0
    total_samples = 0
    interval_loss = 0.0
    interval_samples = 0

    global_step = len(train_loader) * (epoch - 1)
    for batch_idx, batch in enumerate(train_loader, start=1):
        images = batch["image"].to(device)
        labels = batch["class_id"].to(device)

        optimizer.zero_grad()
        logits = network(images)
        losses = criterion(logits, labels)
        loss = losses.mean()
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += losses.sum().item()
        total_samples += batch_size

        interval_loss += losses.sum().item()
        interval_samples += batch_size
        if logger is not None and logging_interval is not None and batch_idx % logging_interval == 0:
            logger.log_metric(
                "train/loss_step",
                interval_loss / interval_samples,
                step=batch_idx + global_step
            )
            interval_loss = 0.0
            interval_samples = 0
    if logger is not None and logging_interval is not None and interval_samples > 0:
        logger.log_metric(
            "train/loss_step",
            interval_loss / interval_samples,
            step=len(train_loader) + global_step
        )
    epoch_loss = total_loss / total_samples
    if logger is not None:
        logger.log_metric(
            "train/loss_epoch",
            epoch_loss,
            step=epoch
        )
    return epoch_loss

@torch.no_grad()
def val_epoch(
    network,
    val_loader,
    criterion,
    device=DEVICE
):
    network.eval()

    total_loss = 0.0
    total_samples = 0

    predictions = []
    class_ids = []
    context_ids = []
    group_ids = []
    for batch in val_loader:
        images = batch["image"].to(device)
        labels = batch["class_id"].to(device)

        logits = network(images)
        losses = criterion(logits, labels)

        predict = logits.argmax(dim=1)

        batch_size = images.shape[0]
        total_loss += losses.sum().item()
        total_samples += batch_size

        predictions.extend(predict.cpu().tolist())
        class_ids.extend(labels.cpu().tolist())
        context_ids.extend(batch["context_id"].tolist())
        group_ids.extend(batch["group_id"].tolist())
    
    val_loss = total_loss / total_samples

    val_metrics, group_accuracy, class_accuracy = metrics(
        predictions=predictions,
        class_ids=class_ids,
        context_ids=context_ids,
        group_ids=group_ids
    )

    return val_loss, val_metrics, group_accuracy, class_accuracy


def train_val(
    network,
    n_epochs,
    criterion,
    optimizer,
    data_loader,
    model_selection,
    config,
    device=DEVICE,
    logging_interval: Optional[int] = None,
    log_dir: Optional[str] = None,
    early_stopping_patience: Optional[int] = None,
    verbose: bool = False
):
    train_loader, val_loader, test_loader = data_loader
    data_dir = train_loader.dataset.data_dir

    logger = None
    experiment_name = config["experiment_name"]
    if log_dir is not None:
        experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)

        logger = comet_ml.start(
            project_name=config["project_name"],
            experiment_config=comet_ml.ExperimentConfig(
                name=experiment_name,
                offline_directory=experiment_dir
            ),
            online=config["logging_online"]
        )
        logger.log_parameters(config)

        log_dataset_distribution(
            dataset=train_loader.dataset,
            logger=logger,
            split="train",
        )
        log_dataset_distribution(
            dataset=val_loader.dataset,
            logger=logger,
            split="val",
        )
        log_dataset_distribution(
            dataset=test_loader.dataset,
            logger=logger,
            split="test",
        )
            
    selection_metric = model_selection["metric"]
    selection_mode = model_selection["mode"]

    best_metric = float("-inf") if selection_mode == "max" else float("inf")
    best_epoch = 0

    best_model_path = os.path.join(
        f"{log_dir}/{experiment_name}" if log_dir is not None else f"./{experiment_name}",
        f"best_model.pt"
    )
    for epoch in range(1, n_epochs + 1):
        if verbose:
            print(f"\nEpoch {epoch}/{n_epochs}")
        train_loss = train_epoch(
            network=network,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            logging_interval=logging_interval,
            logger=logger
        )
        val_loss, val_metrics, val_group_accuracy, val_class_accuracy = val_epoch(
            network=network,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )
        if logger is not None:
            logger.log_metric(
                "val/loss",
                val_loss,
                step=epoch
            )
            log_val_metrics(
                split="val",
                epoch=epoch,
                val_metrics=val_metrics,
                group_accuracy=val_group_accuracy,
                class_accuracy=val_class_accuracy,
                data_dir=data_dir,
                logger=logger
            )

        if verbose:
            print(f"Train_loss: {train_loss:.6f}, Val_loss: {val_loss:.6f}")
            for metric_name, metric_value in val_metrics.items():
                print(f"Val_{metric_name}: {metric_value:.6f}")

        selection_value = get_selection_value(val_loss, val_metrics, selection_metric)
        if is_better_metric(selection_value, best_metric, selection_mode):
            best_metric = selection_value
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "selection_metric": selection_metric,
                    "selection_value": best_metric,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics
                },
                best_model_path
            )
        elif early_stopping_patience is not None and (epoch - best_epoch) >= early_stopping_patience:
            break

    checkpoint = torch.load(best_model_path, map_location=device)
    network.load_state_dict(checkpoint["model_state_dict"])

    _, test_metrics, test_group_accuracy, test_class_accuracy = val_epoch(
        network=network,
        val_loader=test_loader,
        criterion=criterion,
        device=device
    )
    if logger is not None:
        log_val_metrics(
            split="test",
            epoch=best_epoch,
            val_metrics=test_metrics,
            group_accuracy=test_group_accuracy,
            class_accuracy=test_class_accuracy,
            data_dir=data_dir,
            logger=logger
        )
        
        logger.log_metric(
            f"best/val_{selection_metric}",
            best_metric,
            step=best_epoch
        )
        logger.log_metric(
            "best/epoch",
            best_epoch,
            step=best_epoch
        )

        logger.end()
    if verbose:
        print(f"\nTest result")
        for metric_name, metric_value in test_metrics.items():
            print(f"{metric_name}: {metric_value:.6f}")
