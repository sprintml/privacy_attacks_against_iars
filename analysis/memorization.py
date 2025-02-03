from omegaconf import open_dict
from hydra import initialize, compose
import sys
import os

import numpy as np
from torch import Tensor as T
import torch

sys.path.append("./VAR")
sys.path.append("./mar")
sys.path.append("./rar")

from src import gen_models
from src.models import GeneralVARWrapper
from src.dataloaders import loaders


import pandas as pd

from torch import Tensor as T
from typing import List
from tqdm import tqdm

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from torchvision.transforms import ToPILImage

set_plt = lambda: plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.size": 15,  # Set font size to 11pt
        "axes.labelsize": 15,  # -> axis labels
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2,
        "text.usetex": False,
        "pgf.rcfonts": False,
    }
)

set_plt()


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


ids = list(range(8))
device = "cuda"


def get_datasets_indices(config, model_cfg, dataset_cfg, ids, split):
    dss = []
    indices = []
    if split == "val":
        with open_dict(dataset_cfg):
            dataset_cfg.split = split
            dataset_cfg.gpus_cnt = 1
            dataset_cfg.idx = 0
        loader = loaders[model_cfg.dataloader](config, model_cfg, dataset_cfg)
        g = torch.Generator()
        g.manual_seed(model_cfg.seed)
        _ = torch.empty((), dtype=torch.int64).random_(generator=g).item()
        dataset = loader.dataset
        dss.append(dataset)
        indices.append(torch.randperm(len(dataset), generator=g)[:140000].unsqueeze(0))
        return dss, torch.cat(indices, dim=0)

    for i in tqdm(ids):
        with open_dict(dataset_cfg):
            dataset_cfg.split = "train"
            dataset_cfg.gpus_cnt = 8
            dataset_cfg.idx = i
        loader = loaders[model_cfg.dataloader](config, model_cfg, dataset_cfg)
        g = torch.Generator()
        g.manual_seed(model_cfg.seed)
        _ = torch.empty((), dtype=torch.int64).random_(generator=g).item()
        dataset = loader.dataset
        dss.append(dataset)
        indices.append(torch.randperm(len(dataset), generator=g)[:140000].unsqueeze(0))
    return dss, torch.cat(indices, dim=0)


def get_sample_from_idx(all_datasets, all_indices, idx):
    dataset_idx = idx // 140000
    sample_idx = idx % 140000
    final_idx = all_indices[dataset_idx, sample_idx].item()
    image, label = all_datasets[dataset_idx][final_idx]
    return image


def tokens_to_img_var(tokens: T, model):
    return (
        model.tokenizer.idxBl_to_img(
            tokens_to_token_list(tokens), same_shape=False, last_one=True
        )
        .add_(1)
        .mul_(0.5)
    )


def tokens_to_img_rar(tokens: T, model):
    img = model.tokenizer.decode_tokens(tokens.view(tokens.shape[0], -1))
    img = torch.clamp(img, 0.0, 1.0)
    return img


def tokens_to_img_mar(tokens: T, model):
    return (
        model.tokenizer.decode(model.generator.unpatchify(tokens) / 0.2325)
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

np.random.seed(0)

mapping = {
    "var_30": ["cosine_4", 4, 428, [4502, 2120, 3055, 2590], [2256]],
    "rar_xxl": ["cosine_30", 4, 2120, [4500, 2120, 3055, 2590], []],
    "mar_h": ["cosine_5", 2, 3616, [], [2256]],
}
ATTACK = {
    "var_16": "mem_info",
    "var_20": "mem_info",
    "var_24": "mem_info",
    "var_30": "mem_info",
    "rar_xxl": "mem_info",
    "mar_b": "mem_info_mar",
    "mar_l": "mem_info_mar",
    "mar_h": "mem_info_mar",
}
out_main = []
out_uni1 = []
out_uni2 = []
out_appendix = []
SPLIT = "train"

cols_mem = {
    "var_30": 4,
    "rar_xxl": 30,
    "mar_h": 5,
}


def plot(
    images: T, ncols: int, nrows: int, titles: List[str], ylabels: List[str], path: str
) -> None:
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 5, nrows * 5),
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )
    for idx, title in enumerate(titles):
        if nrows > 1:
            axs[0, idx].set_title(title)
        else:
            axs[idx].set_title(title)
    for idx, ylabel in enumerate(ylabels):
        if nrows > 1:
            axs[idx, 0].set_ylabel(ylabel, rotation=90, labelpad=10)
        else:
            axs[idx].set_ylabel(ylabel, rotation=90, labelpad=10)
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        ax.imshow(images[i].permute(1, 2, 0))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0.1)


def main():
    os.makedirs("analysis/plots/memorization", exist_ok=True)
    for MODEL in tqdm(mapping.keys()):
        attack = ATTACK[MODEL]
        with initialize(config_path="conf"):
            config = compose("config").cfg
            model_cfg = compose(f"model/{MODEL}").model
            dataset_cfg = compose("dataset/imagenet").dataset

        with open_dict(dataset_cfg):
            dataset_cfg.split = "train"
            dataset_cfg.gpus_cnt = 8
            dataset_cfg.idx = 5

        with open_dict(model_cfg):
            model_cfg.device = "cpu"
            model_cfg.seed = config.seed

        with open_dict(config):
            config.run_id = "1M_2"
            config.n_samples_eval = 140000

        all_datasets, indices = get_datasets_indices(
            config, model_cfg, dataset_cfg, ids, SPLIT
        )
        model: GeneralVARWrapper = gen_models[model_cfg.name](
            config, model_cfg, dataset_cfg
        )
        model.tokenizer.to(device)
        data = np.load(
            f"out/features/{MODEL}_{attack}_memorized_imagenet_{SPLIT}.npz",
            allow_pickle=True,
        )["data"]

        df = pd.read_csv(f"{MODEL}_memorized_train.csv")
        df["Model"] = MODEL
        df = df[
            [
                "Model",
                "sample_idx",
                f"cosine_{cols_mem[MODEL]}",
                f"l2_{cols_mem[MODEL]}",
            ]
        ]
        df.rename(
            columns={
                f"cosine_{cols_mem[MODEL]}": "cosine",
                f"l2_{cols_mem[MODEL]}": "l2",
            },
            inplace=True,
        )
        df = df.loc[df.cosine > 0.75]

        data = torch.from_numpy(data).to(device)
        col, c_idx, s_idx, uni_indices1, uni_indices2 = mapping[MODEL]
        pred = data[[s_idx], c_idx, :-2].to(device)
        out_pred = tokens_to_img[MODEL](pred, model)
        sample = get_sample_from_idx(
            all_datasets, indices, data[[s_idx], 5, -1].long().item()
        )
        if MODEL not in ["rar_xxl"]:
            sample = sample.add_(1).mul_(0.5)
        out_main.append([sample.unsqueeze(0).cpu(), out_pred.cpu()])

        if model not in ["mar_h"]:
            for uni_idx in uni_indices1:
                pred = data[[uni_idx], c_idx, :-2].to(device)
                out_pred = tokens_to_img[MODEL](pred, model)
                sample = get_sample_from_idx(
                    all_datasets, indices, data[[uni_idx], 5, -1].long().item()
                )
                if MODEL not in ["rar_xxl"]:
                    sample = sample.add_(1).mul_(0.5)
                out_uni1.append([sample.unsqueeze(0).cpu(), out_pred.cpu()])

        if model not in ["rar_xxl"]:
            for uni_idx in uni_indices2:
                pred = data[[uni_idx], c_idx, :-2].to(device)
                out_pred = tokens_to_img[MODEL](pred, model)
                sample = get_sample_from_idx(
                    all_datasets, indices, data[[uni_idx], 5, -1].long().item()
                )
                if MODEL not in ["rar_xxl"]:
                    sample = sample.add_(1).mul_(0.5)
                out_uni2.append([sample.unsqueeze(0).cpu(), out_pred.cpu()])

        for idx in np.random.choice(df.index.values, 4, replace=False):
            print(idx)
            pred = data[[idx], c_idx, :-2].to(device)
            out_pred = tokens_to_img[MODEL](pred, model)
            sample = get_sample_from_idx(
                all_datasets, indices, data[[idx], 5, -1].long().item()
            )
            if MODEL not in ["rar_xxl"]:
                sample = sample.add_(1).mul_(0.5)
            out_appendix.append([sample.unsqueeze(0).cpu(), out_pred.cpu()])

    out_appendix_ = []
    for idx in range(4):
        for m_idx in range(3):
            out_appendix_.append(out_appendix[m_idx * 4 + idx])

    out_uni_1 = []
    for idx in range(4):
        out_uni_1.append([out_uni1[idx][0], out_uni1[idx][1], out_uni1[idx + 4][1]])

    print(len(out_uni2))
    out_uni_2 = [out_uni2[0][0], out_uni2[0][1], out_uni2[1][1]]
    print(len(out_uni_2))

    images_teaser = (
        torch.cat(
            [
                torch.cat(
                    [torch.cat([o[0]]) for o in out_main]
                    + [torch.cat([o[1]]) for o in out_main]
                )
            ]
        )
        .detach()
        .cpu()
    )
    plot(
        images=images_teaser,
        ncols=3,
        nrows=2,
        titles=["Var-$\\mathit{d}30$", "RAR-XXL", "MAR-H"],
        ylabels=["Original", "Extracted"],
        path="analysis/plots/memorization/mem_teaser.png",
    )

    images_uni1 = torch.cat([torch.cat(o) for o in out_uni_1]).detach().cpu()
    plot(
        images=images_uni1,
        ncols=3,
        nrows=4,
        titles=["Original", "Var-$\\mathit{d}30$", "RAR-XXL"],
        ylabels=["", "", "", ""],
        path="analysis/plots/memorization/mem_uni1.png",
    )

    images_uni2 = torch.cat(out_uni_2).detach().cpu()
    plot(
        images=images_uni2,
        ncols=3,
        nrows=1,
        titles=["Original", "Var-$\\mathit{d}30$", "MAR-H"],
        ylabels=[""],
        path="analysis/plots/memorization/mem_uni2.png",
    )

    images_appendix = (
        torch.cat([torch.cat([o[0], o[1]]) for o in out_appendix_]).detach().cpu()
    )
    plot(
        images=images_appendix,
        ncols=6,
        nrows=4,
        titles=[
            "Original",
            "Var-$\\mathit{d}30$",
            "Original",
            "RAR-XXL",
            "Original",
            "MAR-H",
        ],
        ylabels=["", "", "", ""],
        path="analysis/plots/memorization/mem_appendix.png",
    )


if __name__ == "__main__":
    main()
