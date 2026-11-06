import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_transform(config):
    new_config = {}
    for key, values in config.items():
        if isinstance(values, dict):
            x = config_transform(values)
            for new_key, new_values in x.items():
                new_config[key + "_" + new_key] = new_values
        elif isinstance(values, (int, float, str, bool)) or values is None:
            new_config[key] = values
        else:
          new_config[key] = str(values)
    return new_config
