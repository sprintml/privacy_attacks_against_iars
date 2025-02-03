from src.attacks import FeatureExtractor
from torch import Tensor as T
from torch.nn import functional as F
import torch

from typing import Tuple
import zlib


class LLMMIAExtractor(FeatureExtractor):
    def get_all_probas_logprobas(self, logits: T) -> Tuple[T, T]:
        probas = F.softmax(logits, dim=2)
        logprobas = F.log_softmax(logits, dim=2)
        return probas, logprobas  # (B, N_tokens, V), (B, N_tokens, V)

    def get_token_probas_logprobas(self, logits: T, tokens: T) -> Tuple[T, T]:
        probas, logprobas = self.get_all_probas_logprobas(logits)
        token_probas = torch.gather(probas, 2, tokens.unsqueeze(2).long())[..., 0]
        token_logprobas = torch.gather(logprobas, 2, tokens.unsqueeze(2).long())[..., 0]

        return token_probas, token_logprobas  # (B, N_tokens), (B, N_tokens)

    def get_sample_losses(self, losses: T) -> T:
        return losses.mean(dim=1).unsqueeze(1)  # B, 1

    def get_zlib_ratios(self, token_logprobas: T, tokens: T) -> T:
        zlib_entropies = (
            torch.tensor(
                [
                    len(zlib.compress(bytes(str(t.cpu().tolist()), "utf=8")))
                    for t in tokens
                ]
            )
            .unsqueeze(1)
            .to(self.device)
        )

        # negative because we want members to have lower values, and logprobas are negative
        return -token_logprobas.mean(dim=1).unsqueeze(1) / zlib_entropies  # B, 1

    def get_min_k_logprobas(self, token_logprobas: T, k: int) -> T:

        # negative because we want members to have lower values
        return (
            torch.topk(-token_logprobas, k, dim=1).values.mean(dim=1).unsqueeze(1)
        )  # B, 1

    def get_mu(self, logits: T) -> T:
        probas, logprobas = self.get_all_probas_logprobas(logits)
        return (probas * logprobas).sum(dim=2)  # B, N_tokens

    def get_sigma(self, logits: T, mu: T) -> T:
        probas, logprobas = self.get_all_probas_logprobas(logits)
        sigma = (probas * torch.square(logprobas)).sum(dim=2) - torch.square(mu)
        return sigma  # B, N_tokens

    def get_min_k_plus(self, token_logprobas: T, mu: T, sigma: T) -> T:
        return (token_logprobas - mu) / (sigma + 1e-6)  # B, N_tokens

    def get_min_k_plus_logprobas(
        self, token_logprobas: T, mu: T, sigma: T, k: int
    ) -> T:
        minkplus = self.get_min_k_plus(token_logprobas, mu, sigma)

        # negative because we want members to have lower values
        return torch.topk(-minkplus, k, dim=1).values.mean(dim=1).unsqueeze(1)  # B, 1

    def get_surp_k_counter_entropy_logprobas(
        self, mu: T, token_logprobas: T, k: int, max_entropy: float
    ) -> T:
        bound_k = (
            torch.topk(-token_logprobas, k, dim=1).values.max(dim=1).values.unsqueeze(1)
        )
        mask = (-mu < max_entropy) & (token_logprobas < bound_k)
        surp_counter_k = mask.float().mean(dim=1)

        # negative because we want members to have lower values
        return -surp_counter_k.unsqueeze(1)  # B, 1

    def get_surp_k_entropy_logprobas(
        self, mu: T, token_logprobas: T, k: int, max_entropy: float
    ) -> T:
        bound_k = (
            torch.topk(-token_logprobas, k, dim=1).values.max(dim=1).values.unsqueeze(1)
        )
        mask = (-mu < max_entropy) & (token_logprobas < bound_k)
        surp_k = (token_logprobas * mask).sum(dim=1) / mask.sum(dim=1).clip(1e-6, None)

        # negative because we want members to have lower values
        return -surp_k.unsqueeze(1)  # B, 1

    def get_below_mean_logproba(self, token_logprobas: T) -> T:

        # positive because we want members to have lower values
        return (
            (token_logprobas < token_logprobas.mean(dim=1).unsqueeze(1))
            .float()
            .mean(dim=1)
            .unsqueeze(1)
        )  # B, 1

    def get_below_mean_min_k_plus_logproba(
        self, token_logprobas: T, mu: T, sigma: T
    ) -> T:
        minkplus = self.get_min_k_plus(token_logprobas, mu, sigma)

        # positive because we want members to have lower values
        return (
            (minkplus < minkplus.mean(dim=1).unsqueeze(1))
            .float()
            .mean(dim=1)
            .unsqueeze(1)
        )  # B, 1

    def get_below_prev_mean_logproba(self, token_logprobas: T) -> T:

        # negative because we want members to have lower values
        return (
            -(
                token_logprobas
                < token_logprobas.cumsum(dim=1)
                / torch.arange(
                    1, token_logprobas.shape[1] + 1, device=self.device
                ).unsqueeze(0)
            )
            .float()
            .mean(dim=1)
            .unsqueeze(1)
        )  # B, 1

    def get_below_prev_mean_min_k_plus(self, token_logprobas: T, mu: T, sigma: T) -> T:
        minkplus = self.get_min_k_plus(token_logprobas, mu, sigma)

        # negative because we want members to have lower values
        return (
            -(
                minkplus
                < minkplus.cumsum(dim=1)
                / torch.arange(1, minkplus.shape[1] + 1, device=self.device).unsqueeze(
                    0
                )
            )
            .float()
            .mean(dim=1)
            .unsqueeze(1)
        )  # B, 1

    def get_slope(self, token_logprobas: T) -> T:
        x = torch.arange(
            token_logprobas.shape[1], dtype=torch.float, device=self.device
        )
        slope_num = (
            (x - x.mean())
            * (token_logprobas - token_logprobas.mean(dim=1).unsqueeze(1))
        ).sum(dim=1)
        slope_den = (x - x.mean()).pow(2).sum()

        # positive because we want members to have lower values
        return (slope_num / slope_den).unsqueeze(1)  # B, 1

    def get_hinge(self, logits: T, tokens: T) -> T:
        B, N, _ = logits.shape
        token_logits = torch.gather(logits, 2, tokens.unsqueeze(2).long())

        logits[torch.arange(B).unsqueeze(1), torch.arange(N), tokens] = float("-inf")

        hinge = token_logits[..., 0] - torch.max(logits, dim=2).values

        # negative because we want members to have lower values
        return -hinge.mean(dim=1).unsqueeze(1)  # B, 1

    def compute_all(self, logits: T, tokens: T, token_losses: T) -> T:
        token_logprobas = self.get_token_probas_logprobas(logits, tokens)[1]
        mu = self.get_mu(logits)
        sigma = self.get_sigma(logits, mu)

        features = []
        if "loss" in self.attack_cfg.metric_list:
            features.append(self.get_sample_losses(token_losses))
            # print("loss", features.shape)
        if "zlib_ratio" in self.attack_cfg.metric_list:
            features.append(self.get_zlib_ratios(token_logprobas, tokens))
            # print("zlib_ratio", features.shape)
        if "hinge" in self.attack_cfg.metric_list:
            features.append(self.get_hinge(logits.clone(), tokens))
            # print("hinge", features.shape)
        if "min_k%" in self.attack_cfg.metric_list:
            for k in self.attack_cfg.ks:
                k = int(k * tokens.shape[1])
                features.append(self.get_min_k_logprobas(token_logprobas, k))
            # print("min_k%", features.shape)
        if "surp" in self.attack_cfg.metric_list:
            for k in self.attack_cfg.ks:
                k = int(k * tokens.shape[1])
                for max_entropy in self.attack_cfg.max_entropies:
                    features.append(
                        self.get_surp_k_entropy_logprobas(
                            mu, token_logprobas, k, max_entropy
                        )
                    )
            for k in self.attack_cfg.ks:
                k = int(k * tokens.shape[1])
                for max_entropy in self.attack_cfg.max_entropies:
                    features.append(
                        self.get_surp_k_counter_entropy_logprobas(
                            mu, token_logprobas, k, max_entropy
                        )
                    )
            # print("surp", features.shape)
        if "min_k%++" in self.attack_cfg.metric_list:
            for k in self.attack_cfg.ks:
                k = int(k * tokens.shape[1])
                features.append(
                    self.get_min_k_plus_logprobas(token_logprobas, mu, sigma, k)
                )
            # print("min_k%++", features.shape)
        if "camia" in self.attack_cfg.metric_list:
            features.append(self.get_below_mean_logproba(token_logprobas))
            features.append(
                self.get_below_mean_min_k_plus_logproba(token_logprobas, mu, sigma)
            )
            features.append(self.get_below_prev_mean_logproba(token_logprobas))
            features.append(
                self.get_below_prev_mean_min_k_plus(token_logprobas, mu, sigma)
            )
            features.append(self.get_slope(token_logprobas))
            # print("camia", features.shape)
        features = torch.stack(features, dim=2).cpu()
        return features  # B, 1, N_features

    def process_batch(self, batch: Tuple[T, T]) -> T:
        images, classes = batch
        images = images.to(self.device)
        if type(classes) == T:
            classes = classes.to(self.device).long()

        tokens = self.model.tokenize(images)
        logits = self.model.forward(images, classes, is_cfg=False)
        if self.attack_cfg.is_cfg:
            logits_uncond = self.model.forward(
                images,
                torch.full_like(
                    classes,
                    fill_value=self.dataset_cfg.num_classes,
                    device=self.model_cfg.device,
                    dtype=torch.long,
                ),
                is_cfg=False,
            )
            logits = logits - logits_uncond
        token_losses = self.model.get_loss_for_tokens(logits, tokens)

        return self.compute_all(logits, tokens, token_losses)
