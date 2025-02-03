from src.attacks.features_extraction.llm_mia import LLMMIAExtractor
from src.attacks.features_extraction.llm_mia_loss import LLMMIALossExtractor
from torch import Tensor as T
import torch

from typing import Tuple


class DefenseExtractor(LLMMIAExtractor):
    def apply_defense(
        self,
        logits: T,
        std: float,
    ) -> T:
        logits = logits + torch.randn_like(logits) * std
        return logits

    def process_batch(self, batch: Tuple[T, T]) -> T:
        images, classes = batch
        images = images.to(self.device)
        if type(classes) == T:
            classes = classes.to(self.device).long()

        tokens = self.model.tokenize(images)
        logits = self.model.forward(images, classes, is_cfg=False)
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

        features = []
        for std in self.attack_cfg.stds:
            logits_def = self.apply_defense(logits.clone(), std)
            logits_uncond_def = self.apply_defense(
                logits_uncond.clone(),
                std,
            )
            logits_def = logits_def - logits_uncond_def

            token_losses = self.model.get_loss_for_tokens(logits_def, tokens)
            features.append(
                self.compute_all(
                    logits_def.clone(), tokens.clone(), token_losses.clone()
                ).cpu()
            )

        return torch.cat(features, dim=1)


class DefenseLossExtractor(LLMMIALossExtractor):
    def apply_defense(
        self,
        losses: T,
        std: float,
    ) -> T:
        losses = losses + torch.norm(
            torch.randn_like(losses.unsqueeze(0), device=self.model_cfg.device) * std,
            p=2,
            dim=0,
        )
        return losses

    def process_batch(self, batch: Tuple[T, T]) -> T:
        images, classes = batch
        images = images.to(self.device)
        if type(classes) == T:
            classes = classes.to(self.device).long()

        idx = self.attack_cfg.idx
        tokens = self.model.tokenize(images[idx * 16 : (idx + 1) * 16])

        timestep = self.attack_cfg.timestep
        token_losses = self.model.get_loss_per_token(
            images[idx * 16 : (idx + 1) * 16],
            classes[idx * 16 : (idx + 1) * 16],
            ltype="latent",
            is_cfg=self.attack_cfg.is_cfg,
            timestep=timestep,
        )

        features = []
        for std in self.attack_cfg.stds:
            token_losses_def = self.apply_defense(token_losses.clone(), std)
            features.append(
                self.compute_all(token_losses_def.clone(), tokens.clone()).cpu()
            )

        return torch.cat(features, dim=1)
