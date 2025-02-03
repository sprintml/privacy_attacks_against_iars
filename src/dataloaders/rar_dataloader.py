import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.local_datasets import datasets


def get_rar_dataloader(
    config: DictConfig, model_cfg: DictConfig, dataset_cfg: DictConfig
):
    t = transforms.Compose(
        [
            transforms.Resize(
                model_cfg.image_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(model_cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ]
    )
    dataset = datasets[dataset_cfg.name](dataset_cfg, t)
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
