from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
import os
from functools import partial

from hashlib import sha256


def is_valid_file(gpus_idx, path: str) -> bool:
    hash = int.from_bytes(sha256(path.encode("utf-8")).digest(), "big")
    gpus_cnt, idx = gpus_idx
    return hash % gpus_cnt == idx


class ImageFolderDataset:
    def __init__(self, dataset_cfg, transform: Compose) -> ImageFolder:
        self.dataset = ImageFolder(
            os.path.join(dataset_cfg.dataset_path, dataset_cfg.split),
            transform=transform,
            is_valid_file=partial(
                is_valid_file, (dataset_cfg.gpus_cnt, dataset_cfg.idx)
            ),
        )
        self.collate_fn = None