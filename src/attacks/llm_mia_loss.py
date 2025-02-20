# https://arxiv.org/abs/2406.06443

from src.attacks import FeatureExtractor
from torch import Tensor as T
import torch
from typing import Tuple
import zlib


class LLMMIALossExtractor(FeatureExtractor):
    def get_sample_losses(self, losses: T) -> T:
        return losses.mean(dim=1).unsqueeze(1)  # B, 1

    def get_max_k_losses(self, losses: T, k: int) -> T:
        # here max instead of min, because losses are positive, while logprobas are negative
        return torch.topk(losses, k, dim=1).values.mean(dim=1).unsqueeze(1)  # B, 1

    def get_zlib_ratios(self, losses: T, tokens: T) -> T:
        zlib_entropies = torch.tensor(
            [len(zlib.compress(bytes(str(t.cpu().tolist()), "utf=8"))) for t in tokens]
        ).to(self.device)
        # positive because we want members to have lower values; for logprobas it is the opposite
        return (losses.mean(dim=1) / zlib_entropies).unsqueeze(1)  # B, 1

    def get_above_mean_losses(self, losses: T) -> T:
        # above, because losses are smaller for members, while logprobas are larger
        return (
            (losses > losses.mean(dim=1).unsqueeze(1)).float().mean(dim=1).unsqueeze(1)
        )  # B, 1

    def get_above_prev_mean_losses(self, losses: T) -> T:
        # above, because losses are smaller for members, while logprobas are larger
        return (
            -(
                losses
                > losses.cumsum(dim=1)
                / torch.arange(1, losses.shape[1] + 1, device=self.device).unsqueeze(0)
            )
            .float()
            .mean(dim=1)
            .unsqueeze(1)
        )  # B, 1

    def get_slope(self, losses: T) -> T:
        x = torch.arange(losses.shape[1], dtype=torch.float, device=self.device)
        slope_num = ((x - x.mean()) * (losses - losses.mean(dim=1).unsqueeze(1))).sum(
            dim=1
        )
        slope_den = (x - x.mean()).pow(2).sum()

        # negative because we want members to have lower values; for logprobas it is the opposite
        return -(slope_num / slope_den).unsqueeze(1)  # B, 1

    def process_batch(self, batch: Tuple[T, T]) -> T:
        """
        TODO: create docstring
        """
        images, classes = batch
        images = images.to(self.device)
        if type(classes) == T:
            classes = classes.to(self.device).long()

        tokens = self.model.tokenize(images)

        try:
            timestep = self.attack_cfg.timestep
        except:
            timestep = None
        token_losses = self.model.get_loss_per_token(
            images, classes, ltype="latent", is_cfg=self.attack_cfg.is_cfg, timestep= timestep,
        )
        features = []
        if "loss" in self.attack_cfg.metric_list:
            features.append(self.get_sample_losses(token_losses))
        if "zlib_ratio" in self.attack_cfg.metric_list:
            features.append(self.get_zlib_ratios(token_losses, tokens))
        if "min_k%" in self.attack_cfg.metric_list:
            for k in self.attack_cfg.ks:
                k = int(k * token_losses.shape[1])
                features.append(self.get_max_k_losses(token_losses, k))
        if "camia" in self.attack_cfg.metric_list:
            features.append(self.get_above_mean_losses(token_losses))
            features.append(self.get_above_prev_mean_losses(token_losses))
            features.append(self.get_slope(token_losses))
        return torch.stack(features, dim=2).cpu()  # B, 1, N_features


LLMMIALossCFGExtractor = LLMMIALossExtractor