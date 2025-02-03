from src.attacks import FeatureExtractor
from torch import Tensor as T
import torch
from typing import Tuple


class MemInfoMARExtractor(FeatureExtractor):
    def process_batch(self, batch: Tuple[T, T]) -> T:
        """
        TODO: create docstring
        """
        images, classes = batch
        B = images.shape[0]
        images = images.to(self.device)
        if type(classes) == T:
            classes = classes.to(self.device).long()

        tokens = self.model.tokenize(images, flatten=True)
        z, _, mask = self.model.forward(
            images, classes, is_cfg=False, return_target_mask=True
        )
        z = z[mask != 0]
        tokens_in = tokens[mask == 0].view(B, -1, tokens.shape[2])
        tokens_pred = self.model.sample(z, B)

        features = torch.cat(
            [tokens, tokens_in, tokens_pred],
            dim=1,
        ).permute(
            0, 2, 1
        )  # B, N_features, N_tokens

        features = torch.cat(
            [features, classes.reshape(B, 1, 1).repeat(1, features.shape[1], 1)], dim=2
        )
        return features.cpu()
