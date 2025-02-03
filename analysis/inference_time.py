from omegaconf import open_dict
from hydra import initialize, compose
import sys
from tqdm import tqdm

sys.path.append("./VAR")
sys.path.append("./mar")
sys.path.append("./rar")

from src import gen_models
from src.models import GeneralVARWrapper

device = "cuda:0"
do_try = False

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

gen_times = dict()

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

    try:
        prof = model.get_seconds_per_image()
    except:
        print(gen_times)
        raise Exception(f"Failed for {model_name}")

    gen_times[model_name] = prof

print(gen_times)
