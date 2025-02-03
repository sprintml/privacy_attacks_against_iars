from omegaconf import open_dict
from hydra import initialize, compose
import sys

import numpy as np
from torch import Tensor as T
from torch.utils.data import DataLoader, TensorDataset
import torch

sys.path.append("./VAR")
sys.path.append("./mar")
sys.path.append("./rar")
import pandas as pd

from src import gen_models
from src.models import GeneralVARWrapper

from torchvision.transforms import transforms
from torch.nn import functional as F

from torch import Tensor as T
from typing import List
from tqdm import tqdm

from torchmetrics.functional import pairwise_cosine_similarity

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rar_xxl")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--std", type=float, default=0.0)
    return parser.parse_args()


args = parse_args()

SPLIT = args.split
MODEL = args.model
device = "cuda"
top_tokens = {
    "var_16": list(range(5)),
    "var_20": list(range(5)),
    "var_24": list(range(5)),
    "var_30": list(range(5)),
    "rar_xxl": [0, 1, 5, 14, 30],
    "mar_b": list(range(5)),
    "mar_l": list(range(5)),
    "mar_h": [0, 1, 5, 14, 30],
}
TOP_TOKENS = top_tokens[MODEL]
ATTACK = {
    "var_16": "mem_info",
    "var_20": "mem_info",
    "var_24": "mem_info",
    "var_30": "mem_info",
    "rar_xxl": "mem_info",
    "mar_b": "mem_info_mar",
    "mar_l": "mem_info_mar",
    "mar_h": "mem_info_mar",
}[MODEL]


@torch.no_grad()
def get_features(img: T) -> T:
    model = torch.jit.load("sscd_disc_mixup.torchscript.pt").to(device).eval()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    skew_320 = transforms.Compose(
        [
            transforms.Resize([320, 320]),
            normalize,
        ]
    )
    features = model(skew_320(img))
    return features


def get_cosine(features_real, features_generated):
    cosine_similarity = (
        pairwise_cosine_similarity(features_real, features_generated)
        .median(0)
        .values.detach()
        .cpu()
    )
    return cosine_similarity


@torch.no_grad()
def max_tile_distance_batch(images1, images2, tile_size=32):
    N, C, H, W = images1.shape
    num_tiles = H // tile_size
    max_distances = torch.zeros(N, device=images1.device)
    for i in range(num_tiles):
        for j in range(num_tiles):
            tile1 = (
                images1[
                    :,
                    :,
                    i * tile_size : (i + 1) * tile_size,
                    j * tile_size : (j + 1) * tile_size,
                ]
                * 255
            )
            tile2 = (
                images2[
                    :,
                    :,
                    i * tile_size : (i + 1) * tile_size,
                    j * tile_size : (j + 1) * tile_size,
                ]
                * 255
            )
            distances = (
                F.mse_loss(tile1, tile2, reduction="none").sum(dim=[1, 2, 3]).sqrt()
                / tile_size**2
            )
            max_distances = torch.maximum(max_distances, distances)

    return max_distances


def tokens_to_token_list(tokens: T) -> List[T]:
    token_list = []
    idx = 0

    patches = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    ps = np.array([patch**2 for patch in patches])
    for patch in ps:
        scale_target = tokens[:, idx : idx + patch] if idx else tokens[:, [0]]
        token_list.append(scale_target)
        idx += patch
    return token_list


with initialize(config_path="conf"):
    config = compose("config").cfg
    model_cfg = compose(f"model/{MODEL}").model
    dataset_cfg = compose("dataset/imagenet").dataset

with open_dict(dataset_cfg):
    dataset_cfg.split = "train"
    dataset_cfg.gpus_cnt = 8
    dataset_cfg.idx = 5

with open_dict(model_cfg):
    model_cfg.device = device
    model_cfg.seed = config.seed

with open_dict(config):
    config.run_id = "1M_2"
    config.n_samples_eval = 140000


def tokens_to_img_var(tokens: T, model, std):
    return (
        model.tokenizer.idxBl_to_img(
            tokens_to_token_list(tokens), same_shape=False, last_one=True
        )
        .add_(1)
        .mul_(0.5)
        .clamp(0, 1)
    )


def tokens_to_img_rar(tokens: T, model, std):
    img = model.tokenizer.decode_tokens(tokens.view(tokens.shape[0], -1))
    img = torch.clamp(img, 0.0, 1.0)
    return img


def tokens_to_img_mar(tokens: T, model, std: float):
    t = tokens.clone()
    t += torch.randn_like(t) * std
    return (
        model.tokenizer.decode(
            model.generator.unpatchify(t.reshape(t.shape[0], 16, 256)) / 0.2325
        )
        .add(1)
        .mul(0.5)
        .clamp(0, 1)
    )


tokens_to_img = {
    "var_16": tokens_to_img_var,
    "var_20": tokens_to_img_var,
    "var_24": tokens_to_img_var,
    "var_30": tokens_to_img_var,
    "rar_xxl": tokens_to_img_rar,
    "mar_b": tokens_to_img_mar,
    "mar_l": tokens_to_img_mar,
    "mar_h": tokens_to_img_mar,
}


def mar_distance(target, pred):
    return torch.sqrt(torch.pow(target - pred, 2).sum(dim=1)).cpu()


distance = {
    "var_16": lambda target, pred: (target == pred).sum(dim=1).cpu(),
    "var_20": lambda target, pred: (target == pred).sum(dim=1).cpu(),
    "var_24": lambda target, pred: (target == pred).sum(dim=1).cpu(),
    "var_30": lambda target, pred: (target == pred).sum(dim=1).cpu(),
    "rar_xxl": lambda target, pred: (target == pred).sum(dim=1).cpu(),
    "mar_b": mar_distance,
    "mar_l": mar_distance,
    "mar_h": mar_distance,
}


@torch.no_grad()
def get_data():
    model: GeneralVARWrapper = gen_models[model_cfg.name](
        config, model_cfg, dataset_cfg
    )
    device = model.model_cfg.device

    if model_cfg.name == "mar_h":
        data = np.load(
            f"out/features/{MODEL}_{ATTACK}_memorized_imagenet_{SPLIT}.npz",
            allow_pickle=True,
        )["data"]
        print(
            f"out/features/{MODEL}_{ATTACK}_memorized_imagenet_{SPLIT}.npz",
            "loaded, shape:",
            data.shape,
        )
    else:
        data = np.load(
            f"out/features/0_{MODEL}_{ATTACK}_memorized_imagenet_{SPLIT}_{args.std}.npz",
            allow_pickle=True,
        )["data"]

        print(
            f"out/features/0_{MODEL}_{ATTACK}_memorized_imagenet_{SPLIT}_{args.std}.npz",
            "loaded, shape:",
            data.shape,
        )

    data = torch.from_numpy(data)
    data = data.reshape(data.shape[0], 6, -1)
    dataset = TensorDataset(data)
    loader = DataLoader(
        dataset,
        num_workers=config.dataloader_num_workers,
        batch_size=model_cfg.batch_size,
        shuffle=False,
    )
    out = []
    for idx, batch in tqdm(enumerate(loader), total=len(loader)):
        B = batch[0].shape[0]
        batch = batch[0].to(device)
        target = batch[:, 5, :-2]
        preds = [batch[:, idx, :-2] for idx in range(5)]
        out_preds = [tokens_to_img[MODEL](pred, model, 0.0).cpu() for pred in preds]
        out_target = tokens_to_img[MODEL](target, model, args.std).cpu()
        features_real = get_features(out_target.to(device).clone()).cpu()
        features_generated = [
            get_features(pred.to(device).clone()).cpu() for pred in out_preds
        ]
        cosines = torch.cat(
            [
                torch.stack(
                    [
                        get_cosine(features_real[[i]], features_pred[[i]]).cpu()
                        for i in range(B)
                    ],
                    dim=0,
                )
                for features_pred in features_generated
            ],
            dim=1,
        ).T
        l2s = torch.cat(
            [
                max_tile_distance_batch(out_target, pred).cpu().unsqueeze(1)
                for pred in out_preds
            ],
            dim=1,
        ).T
        out.append(
            [
                batch[:, 5, -1].cpu(),
                batch[:, 5, -2].cpu(),
                *[distance[MODEL](target, pred) for pred in preds],
                *cosines.tolist(),
                *l2s.tolist(),
            ]
        )

    out = np.concatenate(out, axis=1).T
    print(out.shape)

    df = pd.DataFrame(
        out,
        columns=[
            "sample_idx",
            "label",
            *[f"token_eq_{i}" for i in TOP_TOKENS],
            *[f"cosine_{i}" for i in TOP_TOKENS],
            *[f"l2_{i}" for i in TOP_TOKENS],
        ],
    )
    df.to_csv(f"{MODEL}_memorized_{SPLIT}_{args.std}.csv", index=False)


def main():
    get_data()


if __name__ == "__main__":
    main()
