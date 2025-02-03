import torch
from torch import nn
from torch import Tensor as T

from omegaconf import DictConfig

from typing import List, Tuple


class GeneralVARWrapper(nn.Module):
    def __init__(
        self, config: DictConfig, model_cfg: DictConfig, dataset_cfg: DictConfig
    ):
        super(GeneralVARWrapper, self).__init__()
        self.config = config
        self.model_cfg = model_cfg
        self.dataset_cfg = dataset_cfg

        self.generator, self.tokenizer = self.load_models()
        total_params = sum(p.numel() for p in self.generator.parameters())
        print(f"TOTAL PARAMS:", total_params)

    def load_models(self):
        """
        Loads the generator and tokenizer models
        """
        raise NotImplementedError

    def tokenize(self, images: T) -> T:
        """
        Tokenizes the images, return tensor of shape (batch_size, seq_len)
        """
        raise NotImplementedError

    def forward(self, images: T, conditioning: T) -> T:
        """
        Computes logits of all tokens, returns tensor of shape (batch_size, seq_len, vocab_size)
        """
        raise NotImplementedError

    @torch.no_grad()
    def get_token_list(self, images: T, *args, **kwargs) -> List[T]:
        """
        Returns list of tokens, each tensor of shape (batch_size, n_tokens -- may vary)
        """
        raise NotImplementedError

    def tokens_to_token_list(self, tokens: T) -> List[T]:
        raise NotImplementedError

    def get_memorization_scores(self, members_features: T, ft_idx: int) -> T:
        raise NotImplementedError

    @torch.no_grad()
    def get_loss_per_token(self, images: T, classes: T, *args, **kwargs) -> T:
        """
        Computes the loss per token, returns tensor of shape (batch_size, seq_len)
        """
        tokens = self.tokenize(images)
        logits = self.forward(images, classes)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        return loss_fn(logits.permute(0, 2, 1), tokens)  # B, N_tokens

    @torch.no_grad()
    def sample(
        self,
        folder: str,
        n_samples_per_class: int = 10,
        std: float = 0,
        clamp_min: float = float("-inf"),
        clamp_max: float = float("inf"),
    ) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def generate_single_memorization(
        self, top: int, target_token_list: List[T], label: T, std: float
    ) -> T:
        raise NotImplementedError

    @torch.no_grad()
    def tokens_to_img(self, tokens: T, *args, **kwargs) -> T:
        raise NotImplementedError

    def get_target_label_memorization(
        self, members_features: T, scores: T, sample_classes: T, cls: int, k: int
    ) -> Tuple[T, T, T]:
        raise NotImplementedError

    @torch.no_grad()
    def get_loss_for_tokens(self, preds: T, ground_truth: T) -> T:
        raise NotImplementedError

    @torch.no_grad()
    def get_flops_forward_train(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def get_flops_generate(self) -> int:
        raise NotImplementedError

    @torch.no_grad()
    def get_seconds_per_image(self) -> float:
        raise NotImplementedError


class GeneralVARProfiler(nn.Module):
    def __init__(self, generator, tokenizer, model_cfg):
        super().__init__()
        self.generator = generator
        self.tokenizer = tokenizer
        self.model_cfg = model_cfg

        self.generator.eval()
        self.tokenizer.eval()

    @torch.no_grad()
    def forward(self, is_train: bool, *args, **kwargs) -> T:
        if is_train:
            return self.forward_train(*args, **kwargs)
        return self.forward_generate(*args, **kwargs)

    @torch.no_grad()
    def forward_train(self, *args, **kwargs) -> T:
        raise NotImplementedError

    @torch.no_grad()
    def forward_generate(self, *args, **kwargs) -> T:
        raise NotImplementedError
