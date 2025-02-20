from src.attacks import FeatureExtractor
from torch import Tensor as T
import torch

import numpy as np
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd

from torchvision.transforms import transforms
from torch.nn import functional as F

from tqdm import tqdm
from torchmetrics.functional import pairwise_cosine_similarity


class ExtractMemorized(FeatureExtractor):
    top_tokens = {
        "var_30": list(range(5)),
        "rar_xxl": [0, 1, 5, 14, 30],
        "mar_h": [0, 1, 5, 14, 30],
    }
    ATTACKS = {
        "var_30": "mem_info",
        "rar_xxl": "mem_info",
        "mar_h": "mem_info_mar",
    }


    distance = {
        "var_30": lambda target, pred: (target == pred).sum(dim=1).cpu(),
        "rar_xxl": lambda target, pred: (target == pred).sum(dim=1).cpu(),
        "mar_h": lambda target, pred: torch.sqrt(torch.pow(target - pred, 2).sum(dim=1)).cpu(),
    }

    @torch.no_grad()
    def get_features(self, img: T) -> T:
        model = (
            torch.jit.load("sscd_disc_mixup.torchscript.pt")
            .to(self.model_cfg.device)
            .eval()
        )

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

    def get_cosine(self, features_real: T, features_generated: T) -> T:
        cosine_similarity = (
            pairwise_cosine_similarity(features_real, features_generated)
            .median(0)
            .values.detach()
            .cpu()
        )
        return cosine_similarity

    @torch.no_grad()
    def load_candidates(self) -> DataLoader:
        data = np.load(
            f"out/features/{self.model_cfg.name}_{self.ATTACKS[self.model_cfg.name]}_memorized_imagenet_{self.dataset_cfg.name}_{self.attack_cfg.std}.npz",
            allow_pickle=True,
        )["data"]

        data = torch.from_numpy(data)
        data = data.reshape(data.shape[0], 6, -1)
        dataset = TensorDataset(data)
        loader = DataLoader(
            dataset,
            num_workers=self.config.dataloader_num_workers,
            batch_size=self.model_cfg.batch_size,
            shuffle=False,
        )
        return loader

    def run(self, *args, **kwargs) -> None:
        loader = self.load_candidates()
        device = self.model_cfg.device
        out = []
        for _, batch in tqdm(enumerate(loader), total=len(loader)):
            B = batch[0].shape[0]
            batch = batch[0].to(device)
            target = batch[:, 5, :-2]
            preds = [batch[:, idx, :-2] for idx in range(5)]
            out_preds = [self.model.tokens_to_img(pred, 0.0).cpu() for pred in preds]
            out_target = self.model.tokens_to_img(target, self.attack_cfg.std).cpu()
            features_real = self.get_features(out_target.to(device).clone()).cpu()
            features_generated = [
                self.get_features(pred.to(device).clone()).cpu() for pred in out_preds
            ]
            cosines = torch.cat(
                [
                    torch.stack(
                        [
                            self.get_cosine(
                                features_real[[i]], features_pred[[i]]
                            ).cpu()
                            for i in range(B)
                        ],
                        dim=0,
                    )
                    for features_pred in features_generated
                ],
                dim=1,
            ).T
            out.append(
                [
                    batch[:, 5, -1].cpu(),
                    batch[:, 5, -2].cpu(),
                    *[self.distance[self.model_cfg.name](target, pred) for pred in preds],
                    *cosines.tolist(),
                ]
            )

        out = np.concatenate(out, axis=1).T
        print(out.shape)
        TOP_TOKENS = self.top_tokens[self.model_cfg.name]

        df = pd.DataFrame(
            out,
            columns=[
                "sample_idx",
                "label",
                *[f"token_eq_{i}" for i in TOP_TOKENS],
                *[f"cosine_{i}" for i in TOP_TOKENS],
            ],
        )
        df.to_csv(
            f"analysis/plots/memorization/{self.model_cfg.name}_memorized_{self.model_cfg.name}_{self.attack_cfg.std}.csv",
            index=False,
        )
