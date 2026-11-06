import numpy as np


NUM_CLASSES = 60
NUM_CONTEXTS = 6


def metrics(
    predictions,
    class_ids,
    context_ids,
    group_ids,
):
    predictions = np.asarray(predictions)
    class_ids = np.asarray(class_ids)
    context_ids = np.asarray(context_ids)
    group_ids = np.asarray(group_ids)

    correct = predictions == class_ids
    overall_accuracy = correct.mean()

    group_accuracy = np.full((NUM_CONTEXTS, NUM_CLASSES), np.nan, dtype=np.float64)
    group_accs = []
    group_precs = []
    for group_id in np.unique(group_ids):
        mask_g = group_ids == group_id

        class_id = class_ids[mask_g][0]
        context_id = context_ids[mask_g][0]

        group_acc = correct[mask_g].mean()

        group_accs.append(group_acc)
        group_accuracy[context_id, class_id] = group_acc

        mask_pred = (
            (predictions == class_id)
            & (context_ids == context_id)
        )

        if mask_pred.sum() > 0:
            group_precs.append(
                correct[mask_pred].mean()
            )
        else:
            group_precs.append(0.0)

    group_accs = np.asarray(group_accs)
    group_precs = np.asarray(group_precs)

    average_group_accuracy = group_accs.mean()
    worst_group_accuracy = group_accs.min()

    average_group_precision = group_precs.mean()
    worst_group_precision = group_precs.min()

    class_accuracy = np.full(NUM_CLASSES, np.nan, dtype=np.float64)
    class_accs = []
    class_precs = []
    for class_id in np.unique(class_ids):
        mask_c = class_ids == class_id

        class_acc = correct[mask_c].mean()

        class_accs.append(class_acc)
        class_accuracy[class_id] = class_acc

        mask_pred = predictions == class_id

        if mask_pred.sum() > 0:
            class_precs.append(
                (class_ids[mask_pred] == class_id).mean()
            )
        else:
            class_precs.append(0.0)

    class_accs = np.asarray(class_accs)
    class_precs = np.asarray(class_precs)

    average_class_accuracy = class_accs.mean()
    worst_class_accuracy = class_accs.min()

    average_class_precision = class_precs.mean()
    worst_class_precision = class_precs.min()

    metric_values = {
        "OverallAccuracy": float(overall_accuracy),
        "AverageGroupAccuracy": float(average_group_accuracy),
        "WorstGroupAccuracy": float(worst_group_accuracy),
        "AverageGroupPrecision": float(average_group_precision),
        "WorstGroupPrecision": float(worst_group_precision),
        "AverageClassAccuracy": float(average_class_accuracy),
        "WorstClassAccuracy": float(worst_class_accuracy),
        "AverageClassPrecision": float(average_class_precision),
        "WorstClassPrecision": float(worst_class_precision),
    }
    return metric_values, group_accuracy, class_accuracy


def get_selection_value(
    val_loss,
    val_metrics,
    metric_name: str
):
    if metric_name == "loss":
        return val_loss

    return val_metrics[metric_name]

def is_better_metric(
    current_value: float,
    best_value: float,
    mode: str
) -> bool:
    if mode == "max":
        return current_value > best_value
    if mode == "min":
        return current_value < best_value
