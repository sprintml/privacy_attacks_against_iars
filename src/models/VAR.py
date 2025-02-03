from VAR.models import build_vae_var, VQVAE, VAR
from VAR.models.helpers import sample_with_top_k_top_p_
from src.models.GeneralVARWrapper import GeneralVARWrapper, GeneralVARProfiler
import torch
from torch import Tensor as T
from torch.nn import functional as F
from torchprofile import profile_macs
from torchvision import transforms
from time import time
from functools import partial
from tqdm import tqdm
import os

from typing import List


class VARWrapper(GeneralVARWrapper):
    def load_models(self):
        vae, var = build_vae_var(
            V=self.model_cfg.V,
            Cvae=self.model_cfg.Cvae,
            ch=self.model_cfg.ch,
            share_quant_resi=self.model_cfg.share_quant_resi,
            device=self.model_cfg.device,
            patch_nums=self.model_cfg.patch_nums,
            num_classes=self.dataset_cfg.num_classes,
            depth=self.model_cfg.model_depth,
            shared_aln=self.model_cfg.shared_aln,
        )
        hf_home = "https://huggingface.co/FoundationVision/var/resolve/main"
        vae_ckpt, var_ckpt = (
            "vae_ch160v4096z32.pth",
            f"var_d{self.model_cfg.model_depth}.pth",
        )
        if not os.path.exists(f"model_checkpoints/var/{vae_ckpt}"):
            os.system(f"wget {hf_home}/{vae_ckpt} -O model_checkpoints/var/{vae_ckpt}")
        if not os.path.exists(f"model_checkpoints/var/{var_ckpt}"):
            os.system(f"wget {hf_home}/{var_ckpt} -O model_checkpoints/var/{var_ckpt}")

        vae.load_state_dict(
            torch.load(f"model_checkpoints/var/{vae_ckpt}", map_location="cpu")
        )
        var.load_state_dict(
            torch.load(f"model_checkpoints/var/{var_ckpt}", map_location="cpu")
        )

        vae.eval()
        var.eval()

        vae.to(self.model_cfg.device)
        var.to(self.model_cfg.device)

        return var, vae

    @torch.no_grad()
    def get_token_list(self, images: T, *args, **kwargs) -> List[T]:
        self.tokenizer: VQVAE
        return self.tokenizer.img_to_idxBl(
            images, v_patch_nums=self.model_cfg.patch_nums
        )  # List[B, N_tokens]

    @torch.no_grad()
    def tokenize(self, images: T) -> T:
        return torch.cat(self.get_token_list(images), dim=1)  # B, N_tokens

    @torch.no_grad()
    def forward(self, images: T, conditioning: T, is_cfg: bool) -> T:
        self.generator: VAR
        token_l = self.get_token_list(images)
        var_input = self.tokenizer.quantize.idxBl_to_var_input(token_l)
        out = self.generator(conditioning, var_input)
        if is_cfg:
            conditioning = torch.full_like(
                conditioning, fill_value=self.dataset_cfg.num_classes
            )
            out_cfg = self.generator(conditioning, var_input)
            out = out_cfg - out
        return out  # B, N_tokens, V

    @torch.no_grad()
    def sequential_predict(
        self, images: T, conditioning: T, num_samplings: int, *args, **kwargs
    ) -> List[T]:
        gen = torch.Generator(device=self.model_cfg.device)
        gen.manual_seed(self.model_cfg.seed)

        logits = self.forward(images, conditioning)[:, 1:, :]  # rm condition token
        tokens = sample_with_top_k_top_p_(
            logits_BlV=logits,
            top_k=self.model_cfg.top_k,
            top_p=self.model_cfg.top_p,
            num_samples=num_samplings,
            rng=gen,
        ).permute(
            0, 2, 1
        )  # B, num_samples, N_tokens
        idx = 0
        token_list = []
        for cnt in self.model_cfg.patch_nums:
            token_list.append(tokens[..., idx : idx + cnt**2])
            idx += cnt
        return token_list  # List[B, num_samples, N_tokens]

    @torch.no_grad()
    def tokens_to_latent(self, tokens: T) -> T:
        B = tokens.shape[0]
        num_samples = tokens.shape[1]
        k = round(tokens.shape[2] ** 0.5)

        return (
            self.tokenizer.quantize.embedding(tokens.reshape(B * num_samples, -1))
            .transpose(1, 2)
            .reshape(B, num_samples, self.tokenizer.Cvae, k, k)
        )  # B, num_samples, Cvae, k, k

    @torch.no_grad()
    def images_to_tokenwise_latent(self, images: T) -> T:
        SN = len(self.model_cfg.patch_nums)

        latents = []
        f: T = self.tokenizer.quant_conv(self.tokenizer.encoder(images))
        B, C, H, W = f.shape
        for si, p in enumerate(self.model_cfg.patch_nums):
            z_NC = (
                F.interpolate(f, size=(p, p), mode="area")
                .permute(0, 2, 3, 1)
                .reshape(-1, C)
                if (si != SN - 1)
                else f.permute(0, 2, 3, 1).reshape(-1, C)
            )
            d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(
                self.tokenizer.quantize.embedding.weight.data.square(),
                dim=1,
                keepdim=False,
            )
            d_no_grad.addmm_(
                z_NC, self.tokenizer.quantize.embedding.weight.data.T, alpha=-2, beta=1
            )  # (B*h*w, vocab_size)
            idx_N = torch.argmin(d_no_grad, dim=1)

            idx_Bhw = idx_N.view(B, p, p)
            h_BChw = (
                F.interpolate(
                    self.tokenizer.quantize.embedding(idx_Bhw).permute(0, 3, 1, 2),
                    size=(H, W),
                    mode="bicubic",
                ).contiguous()
                if (si != SN - 1)
                else self.tokenizer.quantize.embedding(idx_Bhw)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            h_BChw = self.tokenizer.quantize.quant_resi[si / (SN - 1)](h_BChw)
            f.sub_(h_BChw)
            latents.append(z_NC.reshape(B, -1, C))

        return torch.concat(latents, dim=1)  # B, N_tokens, Cvae

    @torch.no_grad()
    def get_loss_for_tokens(self, preds: T, ground_truth: T) -> T:
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        return loss_fn(preds.permute(0, 2, 1), ground_truth)  # B, N_tokens

    @torch.no_grad()
    def get_loss_per_token(
        self, images: T, classes: T, is_cfg: bool, ltype: str = "celoss"
    ) -> T:
        """
        Computes the loss per token, returns tensor of shape (batch_size, seq_len)
        """
        assert ltype in ["celoss", "latent"]
        B = images.shape[0]

        def celoss():
            tokens = self.tokenize(images)
            logits = self.forward(images, classes, is_cfg)
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            return loss_fn(logits.permute(0, 2, 1), tokens)  # B, N_tokens

        def latent():
            latents = self.images_to_tokenwise_latent(images)
            logits = self.forward(images, classes, is_cfg)
            tokens_pred = logits.argmax(dim=2).unsqueeze(1)
            latents_pred = []
            idx = 0
            for p in self.model_cfg.patch_nums:
                latents_pred.append(
                    self.tokens_to_latent(tokens_pred[..., idx : idx + p**2]).reshape(
                        B, -1, self.model_cfg.Cvae
                    )
                )
                idx += p**2
            latents_pred = torch.concat(latents_pred, dim=1)
            return torch.norm(latents - latents_pred, dim=2, p=2).view(
                B, -1
            )  # B, N_tokens

        return celoss() if ltype == "celoss" else latent()

    @torch.no_grad()
    def get_flops_forward_train(self):
        images = torch.randn(1, 3, 256, 256).to(self.model_cfg.device)
        conditioning = torch.randint(0, 1000, (1,)).to(self.model_cfg.device)
        token_l = self.get_token_list(images)
        var_input = self.tokenizer.quantize.idxBl_to_var_input(token_l)
        return profile_macs(self.generator, (conditioning, var_input))

    @torch.no_grad()
    def get_flops_generate(self):
        model_to_profile = VARProfiler(self.generator, self.tokenizer, self.model_cfg)
        return profile_macs(model_to_profile, (False,))

    @torch.no_grad()
    def get_seconds_per_image(self) -> float:
        B = self.model_cfg.batch_size
        times = []
        for _ in range(int(10 / (B / 64))):
            start = time()
            sampled_images = self.generator.autoregressive_infer_cfg(
                B=B,
                label_B=torch.randint(0, 1000, (B,), device=self.model_cfg.device),
            )
            times.append(time() - start)

        return sum(times) / (len(times) * B)

    @torch.no_grad()
    def sample(
        self,
        folder: str,
        n_samples_per_class: int = 10,
        std: float = 0,
        clamp_min: float = float("-inf"),
        clamp_max: float = float("inf"),
    ) -> None:
        def apply_defense(
            logits: T,
            std: float,
            clamp_min: float,
            clamp_max: float,
        ) -> T:
            logits = logits + torch.randn_like(logits) * std
            logits = logits.clamp(min=clamp_min, max=clamp_max)
            return logits

        def get_batches(lst: T, batch_size: int) -> list:
            return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]

        def get_logits(generator: VAR, x: T, cond_BD: T):
            if not isinstance(x, torch.Tensor):
                h, resi = x  # fused_add_norm must be used
                h = resi + self.blocks[-1].drop_path(h)
            else:  # fused_add_norm is not used
                h = x
            logits = generator.head(
                generator.head_nm(h.float(), cond_BD).float()
            ).float()
            return apply_defense(logits, std, clamp_min, clamp_max)

        B = self.model_cfg.batch_size
        classes = torch.arange(1000, device=self.model_cfg.device).repeat_interleave(
            n_samples_per_class
        )

        self.generator.get_logits = partial(get_logits, self.generator)

        images = []
        for class_labels in tqdm(get_batches(classes, B)):
            sampled_images = self.generator.autoregressive_infer_cfg(
                B=class_labels.shape[0],
                label_B=class_labels,
                cfg=1.5,
                top_p=0.96,
                top_k=900,
                more_smooth=False,
            )
            images.append(sampled_images.cpu())

        os.makedirs(self.config.path_to_images, exist_ok=True)
        os.makedirs(f"{self.config.path_to_images}/{folder}", exist_ok=True)

        t = transforms.ToPILImage()
        for i, img in enumerate(torch.cat(images, dim=0)):
            t(img).save(f"{self.config.path_to_images}/{folder}/{i}.png")


class VARProfiler(GeneralVARProfiler):
    @torch.no_grad()
    def forward_generate(self):
        self.generator: VAR
        out = self.generator.autoregressive_infer_cfg(
            B=1,
            label_B=0,
        )
        return out
