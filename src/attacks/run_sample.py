import sys
import os

sys.path.append("./VAR")
sys.path.append("./mar")
sys.path.append("./rar")

import numpy as np
from omegaconf import open_dict
from hydra import initialize, compose

import torch.nn as nn
from torch import Tensor as T
from src.models import GeneralVARWrapper
from typing import Tuple, List

from itertools import product
from tqdm import tqdm
import pandas as pd

from src import (
    get_tpr_fpr,
    get_p_value,
    get_accuracy,
    get_auc,
    load_data,
    load_members_nonmembers_scores,
    DataSource,
    gen_models,
    feature_extractors,
    score_computers,
)

MODELS = ["var_16", "var_20", "var_24", "var_30"]

stds = [0.0, 0.001, 0.01, 0.1, 1.0]
clips = [np.inf, 100, 10, 1, 0.1]


def get_model(model: str, split: str) -> GeneralVARWrapper:
    with initialize(config_path="conf"):
        config = compose("config").cfg
        dataset_cfg = compose("dataset/imagenet").dataset
        model_cfg = compose(f"model/{model}").model
    with open_dict(dataset_cfg):
        dataset_cfg.split = split

    with open_dict(model_cfg):
        model_cfg.device = "cuda"
        model_cfg.seed = config.seed

    model = gen_models[model_cfg.name](config, model_cfg, dataset_cfg)
    return model


def main():
    for model_name in MODELS:
        model = get_model(model_name, "val")
        model.sample(f"{model_name}_full", n_samples_per_class=50)
        break


if __name__ == "__main__":
    main()
