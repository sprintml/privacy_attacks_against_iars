# adjusted from https://github.com/FoundationVision/VAR/blob/main/utils/data.py
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.local_datasets import datasets


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def get_var_dataloader(
    config: DictConfig, model_cfg: DictConfig, dataset_cfg: DictConfig
):
    # build augmentations
    mid_reso = round(
        model_cfg.mid_reso * model_cfg.image_size
    )  # first resize to mid_reso, then crop to final_reso
    final_reso = model_cfg.image_size
    aug = [
        transforms.Resize(
            mid_reso, interpolation=transforms.InterpolationMode.LANCZOS
        ),  # transforms.Resize: resize the shorter edge to mid_reso
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(),
        normalize_01_into_pm1,
    ]
    if model_cfg.hflip:
        aug.insert(0, transforms.RandomHorizontalFlip())
    aug = transforms.Compose(aug)
    dataset = datasets[dataset_cfg.name](dataset_cfg, aug)
    collate_fn = dataset.collate_fn
    dataset = dataset.dataset

    g = torch.Generator()
    g.manual_seed(model_cfg.seed)
    return DataLoader(
        dataset,
        num_workers=config.dataloader_num_workers,
        batch_size=model_cfg.batch_size,
        shuffle=True,
        generator=g,
        collate_fn=collate_fn,
    )
