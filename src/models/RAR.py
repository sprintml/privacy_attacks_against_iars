from rar.demo_util import get_config, get_rar_generator, sample_fn
from rar.utils.train_utils import create_pretrained_tokenizer
from rar.modeling.titok import PretrainedTokenizer
from rar.modeling.rar import RAR

from src.models.utils import ARLoss
from src.models.GeneralVARWrapper import GeneralVARWrapper, GeneralVARProfiler

import torch
from torch import Tensor as T
from torch.nn import functional as F
from huggingface_hub import hf_hub_download
from torchprofile import profile_macs

from time import time

from typing import Union, Tuple


class RARWrapper(GeneralVARWrapper):
    def load_models(self):
        rar_config = get_config("rar/configs/training/generator/rar.yaml")
        rar_config.model.generator.hidden_size = self.model_cfg.hidden_size
        rar_config.model.generator.num_hidden_layers = self.model_cfg.num_hidden_layers
        rar_config.model.generator.num_attention_heads = 16
        rar_config.model.generator.intermediate_size = self.model_cfg.intermediate_size

        rar_config.experiment.generator_checkpoint = (
            f"model_checkpoints/rar/{self.model_cfg.model_size}.bin"
        )
        rar_config.model.vq_model.pretrained_tokenizer_weight = (
            "model_checkpoints/rar/maskgit-vqgan-imagenet-f16-256.bin"
        )

        hf_hub_download(
            repo_id="fun-research/TiTok",
            filename=f"maskgit-vqgan-imagenet-f16-256.bin",
            local_dir="model_checkpoints/rar",
        )
        # download the rar generator weight
        hf_hub_download(
            repo_id="yucornetto/RAR",
            filename=f"{self.model_cfg.model_size}.bin",
            local_dir="model_checkpoints/rar",
        )

        tokenizer = create_pretrained_tokenizer(rar_config)
        generator = get_rar_generator(rar_config)

        tokenizer.eval()
        generator.eval()

        tokenizer.to(self.model_cfg.device)
        generator.to(self.model_cfg.device)

        # A hack to get the config for the loss module
        class Obj:
            pass

        loss_config = Obj()
        loss_config.model = Obj()
        loss_config.model.vq_model = Obj()
        loss_config.model.vq_model.codebook_size = 1024

        self.loss_fn = ARLoss(loss_config, reduction="none")

        return generator, tokenizer

    @torch.no_grad()
    def tokenize(self, images: T) -> T:
        self.tokenizer: PretrainedTokenizer
        return self.tokenizer.encode(images)  # B, N_tokens

    @torch.no_grad()
    def forward(
        self,
        images: T,
        conditioning: T,
        is_cfg: bool,
        return_labels: bool = False,
    ) -> Union[T, Tuple[T, T]]:
        self.generator: RAR
        cond = self.generator.preprocess_condition(conditioning)
        tokens = self.tokenize(images)
        logits, labels = self.generator(tokens, cond, return_labels=True)
        if is_cfg:
            cond = self.generator.get_none_condition(conditioning)
            logits_cfg, _ = self.generator(tokens, cond, return_labels=True)
            logits = logits_cfg - logits
        # first one is the class token, which we don't care about
        if return_labels:
            return logits[:, :-1], labels  # (B, N_tokens, V), (B, N_tokens)

        return logits[:, :-1]  # B, N_tokens, V

    @torch.no_grad()
    def get_loss_per_token(
        self, images: T, conditioning: T, is_cfg: bool, **kwargs
    ) -> T:
        logits, labels = self.forward(images, conditioning, is_cfg, return_labels=True)
        return self.loss_fn(logits, labels)[0]  # B, N_tokens

    @torch.no_grad()
    def get_loss_for_tokens(self, preds: T, ground_truth: T) -> T:
        return self.loss_fn(preds, ground_truth)[0]  # B, N_tokens

    @torch.no_grad()
    def get_flops_forward_train(self) -> int:
        images = torch.randn(1, 3, 256, 256).to(self.model_cfg.device)
        conditioning = torch.randint(0, 1000, (1,)).to(self.model_cfg.device)
        cond = self.generator.preprocess_condition(conditioning)
        tokens = self.tokenize(images)
        return profile_macs(self.generator, (tokens, cond, True))

    @torch.no_grad()
    def get_flops_generate(self) -> int:
        model_to_profile = RARProfiler(self.generator, self.tokenizer, self.model_cfg)
        condition = torch.tensor([0], dtype=torch.long).to(self.model_cfg.device)
        before_gen_flops = profile_macs(model_to_profile, ("before_gen", condition))
        ids = model_to_profile.forward_before_gen(condition)

        self.generator.enable_kv_cache()
        step_flops = []
        for step in range(self.generator.image_seq_len):
            args = (
                "single_step",
                condition,
                3.0,
                1.0,
                3.0,
                step,
                ids,
            )
            flops = profile_macs(model_to_profile, args)
            step_flops.append(flops)
            ids = model_to_profile.forward_single_step(*args[1:])
        self.generator.disable_kv_cache()
        decode_flops = profile_macs(model_to_profile, ("decode", ids))

        return before_gen_flops + sum(step_flops) + decode_flops

    @torch.no_grad()
    def get_seconds_per_image(self) -> float:
        B = self.model_cfg.batch_size
        times = []
        for _ in range(10):
            start = time()
            sampled_images = sample_fn(
                self.generator,
                self.tokenizer,
                labels=torch.randint(0, 1000, (B,), device=self.model_cfg.device),
                device=self.model_cfg.device,
            )
            times.append(time() - start)

        return sum(times) / (len(times) * B)


class RARProfiler(GeneralVARProfiler):
    @torch.no_grad()
    def forward(self, stage: str, *args, **kwargs) -> T:
        self.generator: RAR
        self.tokenizer: PretrainedTokenizer
        if stage == "before_gen":
            return self.forward_before_gen(*args, **kwargs)
        if stage == "single_step":
            return self.forward_single_step(*args, **kwargs)
        if stage == "decode":
            return self.forward_decode(*args, **kwargs)

    @torch.no_grad()
    def forward_before_gen(
        self,
        condition: T,
    ):
        condition = self.generator.preprocess_condition(condition, cond_drop_prob=0.0)
        device = condition.device
        num_samples = condition.shape[0]
        ids = torch.full((num_samples, 0), -1, device=device)

        self.generator.enable_kv_cache()

        return ids

    @torch.no_grad()
    def forward_single_step(
        self,
        condition: T,
        guidance_scale: float,
        randomize_temperature: float,
        guidance_scale_pow: float,
        step: int,
        ids: T,
    ):
        num_samples = condition.shape[0]
        scale_pow = torch.ones((1), device=self.model_cfg.device) * guidance_scale_pow
        scale_step = (
            (
                1
                - torch.cos(
                    ((step / self.generator.image_seq_len) ** scale_pow) * torch.pi
                )
            )
            * 1
            / 2
        )
        cfg_scale = (guidance_scale - 1) * scale_step + 1

        logits = self.generator.forward_fn(
            torch.cat([ids, ids], dim=0),
            torch.cat([condition, self.generator.get_none_condition(condition)], dim=0),
            orders=None,
            is_sampling=True,
        )
        cond_logits, uncond_logits = logits[:num_samples], logits[num_samples:]
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale

        # keep the logit of last token
        logits = logits[:, -1]
        logits = logits / randomize_temperature
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        ids = torch.cat((ids, sampled), dim=-1)
        return ids

    @torch.no_grad()
    def forward_decode(self, generated_tokens: T) -> T:
        generated_image = self.tokenizer.decode_tokens(
            generated_tokens.view(generated_tokens.shape[0], -1)
        )

        generated_image = torch.clamp(generated_image, 0.0, 1.0)
        generated_image = (
            (generated_image * 255.0)
            .permute(0, 2, 3, 1)
            .to("cpu", dtype=torch.uint8)
            .numpy()
        )
        return generated_image
