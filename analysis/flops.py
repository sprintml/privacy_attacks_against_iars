from omegaconf import open_dict
from hydra import initialize, compose
import sys
import torch
from tqdm import tqdm

sys.path.append("./VAR")
sys.path.append("./mar")
sys.path.append("./rar")

from src import gen_models
from src.models import GeneralVARWrapper

device = "cuda"

MODELS = [
    "var_16",
    "var_20",
    "var_24",
    "var_30",
    "mar_b",
    "mar_l",
    "mar_h",
    "rar_b",
    "rar_l",
    "rar_xl",
    "rar_xxl",
]


def main():
    flops_gen = dict()
    flops_forward = dict()

    for model_name in tqdm(MODELS):
        with initialize(config_path="conf"):
            config = compose("config").cfg
            model_cfg = compose(f"model/{model_name}").model
            dataset_cfg = compose("dataset/imagenet").dataset
        with open_dict(dataset_cfg):
            dataset_cfg.split = "train"

        with open_dict(model_cfg):
            model_cfg.device = device
            model_cfg.seed = config.seed

        model: GeneralVARWrapper = gen_models[model_cfg.name](
            config, model_cfg, dataset_cfg
        )

        prof_gen = model.get_flops_generate()
        prof_forward = model.get_flops_forward_train()

        flops_gen[model_name] = prof_gen
        flops_forward[model_name] = prof_forward

    print(f"FLOPS GENERATE: {flops_gen}")
    print(f"FLOPS FORWARD: {flops_forward}")


if __name__ == "__main__":
    main()
