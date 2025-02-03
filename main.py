import sys
import os

sys.path.append("./VAR")
sys.path.append("./mar")
sys.path.append("./rar")


import hydra
import torch
import random
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf, open_dict
from itertools import product

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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    model_cfg = cfg.model
    action_cfg = cfg.action
    attack_cfg = cfg.attack
    dataset_cfg = cfg.dataset
    config = cfg.cfg
    with open_dict(model_cfg):
        model_cfg.device = action_cfg.device
        model_cfg.seed = config.seed

    action_input = [config, model_cfg, action_cfg, attack_cfg, dataset_cfg]

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True

    if action_cfg.name == "features_extraction":
        features_extraction(*action_input)
    elif action_cfg.name == "scores_computation":
        scores_computation(*action_input)
    elif action_cfg.name == "evaluation":
        evaluation(*action_input)
    elif action_cfg.name == "evaluation_bulk":
        evaluation_bulk(*action_input)
    else:
        raise ValueError("Invalid action name")

    print("fin")


def features_extraction(
    config: DictConfig,
    model_cfg: DictConfig,
    action_cfg: DictConfig,
    attack_cfg: DictConfig,
    dataset_cfg: DictConfig,
) -> None:
    model = gen_models[model_cfg.name](config, model_cfg, dataset_cfg)
    extractor = feature_extractors[attack_cfg.name](
        config, model_cfg, action_cfg, attack_cfg, dataset_cfg, model
    )
    extractor.run()


if __name__ == "__main__":
    main()