# https://arxiv.org/abs/2406.11838

from mar.util.download import (
    download_pretrained_vae,
    download_pretrained_marb,
    download_pretrained_marl,
    download_pretrained_marh,
)
from mar.models.mar import MAR, mar_base, mar_huge, mar_large
from mar.models.vae import AutoencoderKL
from src.models.GeneralVARWrapper import GeneralVARWrapper, GeneralVARProfiler

import torch
from torch import Tensor as T

import os
import math
import numpy as np

from typing import Union, Tuple, List
from torchprofile import profile_macs

from time import time

download = {
    "base": download_pretrained_marb,
    "large": download_pretrained_marl,
    "huge": download_pretrained_marh,
}

model = {
    "base": mar_base,
    "large": mar_large,
    "huge": mar_huge,
}


def mask_by_order(mask_len: int, order: int, bsz: int, seq_len: int, device) -> T:
    masking = torch.zeros(bsz, seq_len).to(device)
    masking = torch.scatter(
        masking,
        dim=-1,
        index=order[:, : mask_len.long()],
        src=torch.ones(bsz, seq_len).to(device),
    ).bool()
    return masking


class MARWrapper(GeneralVARWrapper):
    def load_models(self):
        vae_ckpt, mar_ckpt = (
            "kl16.ckpt",
            f"mar_{self.model_cfg.size}/checkpoint-last.pth",
        )
        os.makedirs("model_checkpoints/mar", exist_ok=True)
        if not os.path.exists(f"model_checkpoints/mar/{vae_ckpt}"):
            download_pretrained_vae()
            os.system(
                f"mv pretrained_models/vae/{vae_ckpt} model_checkpoints/mar/{vae_ckpt}"
            )
        if not os.path.exists(f"model_checkpoints/mar/mar_{self.model_cfg.size}.ckpt"):
            download[self.model_cfg.size]()
            os.system(
                f"mv pretrained_models/mar/{mar_ckpt} model_checkpoints/mar/mar_{self.model_cfg.size}.ckpt"
            )
        os.system("rm -rf pretrained_models")

        mar = model[self.model_cfg.size](
            buffer_size=64,
            diffloss_d=self.model_cfg.diffloss_d,
            diffloss_w=self.model_cfg.diffloss_w,
            num_sampling_steps=str(self.model_cfg.num_sampling_steps_diffloss),
        ).to(self.model_cfg.device)

        state_dict = torch.load(
            f"model_checkpoints/mar/mar_{self.model_cfg.size}.ckpt"
        )["model_ema"]
        mar.load_state_dict(state_dict)
        mar.eval()

        vae = (
            AutoencoderKL(
                embed_dim=16,
                ch_mult=(1, 1, 2, 2, 4),
                ckpt_path=f"model_checkpoints/mar/{vae_ckpt}",
            )
            .to(self.model_cfg.device)
            .eval()
        )
        mar.diffusion_batch_mul = self.model_cfg.diffusion_batch_mul
        return mar, vae

    @torch.no_grad()
    def tokenize(self, images: T, flatten: bool = True) -> T:
        self.tokenizer: AutoencoderKL
        B = images.size(0)
        tokens = self.tokenizer.encode(images).sample().mul_(0.2325)
        if flatten:
            return self.generator.patchify(tokens)  # B, N_tokens, embed_dim
        return tokens  # B, 16, 16, 16

    def sample_orders(self, B: int, seed: int = None) -> T:
        if seed == None:
            seed = self.config.seed
        orders = []
        for _ in range(B):
            np.random.seed(seed)
            order = np.array(list(range(self.generator.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).to(self.model_cfg.device).long()
        return orders

    def random_masking(self, x: T, orders: T, mask_rate: float) -> T:
        bsz, seq_len, _ = x.shape
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=self.model_cfg.device)
        mask = torch.scatter(
            mask,
            dim=-1,
            index=orders[:, :num_masked_tokens],
            src=torch.ones(bsz, seq_len, device=self.model_cfg.device),
        )
        return mask

    @torch.no_grad()
    def forward(
        self,
        images: T,
        conditioning: T,
        is_cfg: bool,
        return_target_mask: bool = False,
        seed: int = None,
    ) -> Union[T, Tuple[T, T, T]]:
        self.generator: MAR

        tokens = self.tokenize(images, flatten=False)

        # if is_cfg:
        #     class_embedding = self.generator.fake_latent.repeat(images.size(0), 1)
        # else:
        class_embedding = self.generator.class_emb(conditioning)
        x = self.generator.patchify(tokens)
        gt_latents = x.clone().detach()
        orders = self.sample_orders(B=x.size(0), seed=seed)
        mask = self.random_masking(
            x=x, orders=orders, mask_rate=self.model_cfg.mask_rate
        )
        x = self.generator.forward_mae_encoder(x, mask, class_embedding)
        z = self.generator.forward_mae_decoder(x, mask)
        if return_target_mask:
            return (
                z,
                gt_latents,
                mask,
            )  # (B, N_tokens, embed_dim), (B, 16, 16, 16), (B, N_tokens)
        return z  # B, N_tokens, embed_dim

    @torch.no_grad()
    def get_flops_forward_train(self) -> T:
        images = torch.randn(1, 3, 256, 256).to(self.model_cfg.device)
        conditioning = torch.randint(0, 1000, (1,)).to(self.model_cfg.device)
        tokens = self.tokenize(images, flatten=False)

        class_embedding = self.generator.class_emb(conditioning)
        x = self.generator.patchify(tokens)
        gt_latents = x.clone().detach()
        orders = self.sample_orders(B=x.size(0), seed=self.config.seed)
        mask = self.random_masking(
            x=x, orders=orders, mask_rate=self.model_cfg.mask_rate
        )

        model_to_profile = MARProfiler(self.generator, self.tokenizer, self.model_cfg)

        return profile_macs(
            model_to_profile, (True, x, mask, class_embedding, gt_latents)
        )

    @torch.no_grad()
    def get_flops_generate(self) -> int:
        model_to_profile = MARProfiler(self.generator, self.tokenizer, self.model_cfg)
        bsz = 1
        mask = torch.ones(bsz, self.generator.seq_len).cuda()
        tokens = torch.zeros(
            bsz, self.generator.seq_len, self.generator.token_embed_dim
        ).cuda()
        orders = self.generator.sample_orders(bsz)

        indices = list(range(64))
        class_embedding = self.generator.class_emb(
            torch.tensor([0]).to(self.model_cfg.device)
        )
        step_flops = []
        for step in indices:
            args = (
                step,
                tokens.clone(),
                class_embedding.clone(),
                orders.clone(),
                mask.clone(),
            )
            flops = profile_macs(model_to_profile, (False, *args))
            step_flops.append(flops)
            tokens, mask = model_to_profile.forward_generate(*args)
        return sum(step_flops)

    @torch.no_grad()
    def tokens_to_img(self, tokens: T, std: float = 0) -> T:
        t = tokens.clone()
        t += torch.randn_like(t) * std
        return (
            self.tokenizer.decode(self.generator.unpatchify(t) / 0.2325)
            .add(1)
            .mul(0.5)
            .clamp(0, 1)
        )

    @torch.no_grad()
    def get_memorization_scores(self, members_features: T, *args, **kwargs) -> T:
        np.random.seed(0)
        order = np.array(list(range(256)))
        np.random.shuffle(order)
        tokens_order = np.concat(
            [
                np.arange(256)[~np.isin(np.arange(256), order[: int(256 * 0.95)])],
                np.arange(256)[np.isin(np.arange(256), order[: int(256 * 0.95)])],
            ]
        )
        invert_order = np.empty_like(tokens_order)
        invert_order[tokens_order] = np.arange(256)
        distances = members_features[..., tokens_order] - members_features[..., 256:-1]
        distances = torch.pow(distances, 2).sum(dim=1)
        return -distances.mean(dim=1)

    @torch.no_grad()
    def generate_single_mar(
        self, top: int, target_token_list: T, label: T, *args, **kwargs
    ) -> T:
        B = label.size(0)
        num_steps = 64
        masks, masks_pred = self.get_masks_masks_pred_indices(B, num_steps)
        tokens = torch.zeros(
            B, self.generator.seq_len, self.generator.token_embed_dim
        ).to(self.model_cfg.device)
        class_embedding = self.generator.class_emb(label)

        target = target_token_list.clone().view(B, self.generator.seq_len, -1)

        for step, mask, mask_to_pred in zip(range(num_steps), masks, masks_pred):
            cur_tokens = tokens.clone()

            if step < top:
                cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = target[
                    mask_to_pred.nonzero(as_tuple=True)
                ]
            else:
                x = self.generator.forward_mae_encoder(
                    cur_tokens, mask, class_embedding
                )
                z = self.generator.forward_mae_decoder(x, mask)
                z = z[mask_to_pred.nonzero(as_tuple=True)]

                tokens_pred = self.generator.diffloss.sample(z, 1, 1)
                cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = tokens_pred

            tokens = cur_tokens.clone()

        return tokens.view(B, -1)  # B, seq_len * token_embed_dim

    def tokens_to_token_list(self, tokens: T):
        return tokens

    def get_target_label_memorization(
        self, members_features: T, scores: T, sample_classes: T, cls: int, k: int
    ) -> Tuple[T, T, T]:
        mask = sample_classes == cls
        scores_cls = scores.clone()
        scores_cls[~mask] = -torch.inf
        mem_samples_indices = torch.topk(scores_cls, 10).indices
        mem_sample_idx = mem_samples_indices[k]
        label_B = (
            members_features[mem_sample_idx, :][0, [-1]]
            .to(self.model_cfg.device)
            .long()
        )

        target_tokens = (
            members_features[mem_sample_idx, :][:, :256]
            .permute(1, 0)
            .reshape(1, -1)
            .to(self.model_cfg.device)
        )

        return (
            target_tokens,
            label_B,
            mem_sample_idx.unsqueeze(0).to(self.model_cfg.device),
        )

    @torch.no_grad()
    def loss_fn(self, z: T, target: T, mask: T, timestep: int) -> T:

        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(
            self.generator.diffusion_batch_mul, 1
        )
        z = z.reshape(bsz * seq_len, -1).repeat(self.generator.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz * seq_len).repeat(self.generator.diffusion_batch_mul)

        t = torch.ones(target.shape[0], device=self.model_cfg.device).long() * timestep

        model_kwargs = dict(c=z)
        loss_dict = self.generator.diffloss.train_diffusion.training_losses(
            self.generator.diffloss.net, target, t, model_kwargs
        )
        loss = loss_dict["loss"]
        loss = (loss * mask) / mask.sum()
        return loss

    @torch.no_grad()
    def get_loss_per_token(
        self, images: T, conditioning: T, is_cfg: bool, timestep: int, *args, **kwargs
    ) -> T:
        B = images.size(0)
        losses, totals = [], []
        for i in range(self.model_cfg.repeat):
            z, target, mask = self.forward(
                images,
                conditioning,
                False,
                return_target_mask=True,
                seed=self.config.seed + i,
            )
            loss = (
                self.loss_fn(z=z, target=target, mask=mask, timestep=timestep)
                .reshape(self.generator.diffusion_batch_mul, B, -1)
                .permute(1, 0, 2)
                .mean(dim=1)
            )
            if is_cfg:
                z, target, mask = self.forward(
                    images,
                    conditioning,
                    is_cfg,
                    return_target_mask=True,
                    seed=self.config.seed + i,
                )
                loss_cfg = (
                    self.loss_fn(
                        z=torch.zeros_like(z, device=self.model_cfg.device),
                        target=target,
                        mask=mask,
                    )
                    .reshape(B, self.generator.diffusion_batch_mul, -1)
                    .mean(dim=1)
                )
                loss = loss_cfg - loss
            losses.append(loss)
            totals.append((loss != 0).float())
        loss = torch.stack(losses, dim=2).sum(dim=2)
        total = torch.stack(totals, dim=2).sum(dim=2) + 1e-6  # avoid division by zero

        return loss / total

    @torch.no_grad()
    def tokens_to_latent(self, tokens: T) -> T:
        B = tokens.size(0)
        return tokens.view(B, -1, self.generator.token_embed_dim)

    def get_masks_masks_pred_indices(
        self, B: int, num_steps: int
    ) -> Tuple[List[T], List[T]]:
        mask = torch.ones(B, self.generator.seq_len).to(self.model_cfg.device)
        orders = self.sample_orders(B)
        indices = list(range(num_steps))
        masks, masks_pred = [], []
        for step in indices:
            masks.append(mask)

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / num_steps)
            mask_len = torch.Tensor([np.floor(self.generator.seq_len * mask_ratio)]).to(
                self.model_cfg.device
            )

            # masks out at least one for the next iteration
            mask_len = torch.maximum(
                torch.Tensor([1]).to(self.model_cfg.device),
                torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len),
            )

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(
                mask_len[0],
                orders,
                B,
                self.generator.seq_len,
                device=self.model_cfg.device,
            )
            if step >= num_steps - 1:
                mask_to_pred = mask[:B].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:B].bool(), mask_next.bool())
            mask = mask_next

            masks_pred.append(mask_to_pred)

        return masks, masks_pred

    @torch.no_grad()
    def get_token_list(self, images: T, num_steps: int) -> List[T]:
        B = images.size(0)
        masks = self.get_masks_masks_pred_indices(B, num_steps)[1]
        tokens = self.tokenize(images)
        token_list = []
        for mask in masks:
            token_list.append(
                tokens[mask.nonzero(as_tuple=True)].view(
                    B, 1, -1, self.generator.token_embed_dim
                )
            )
        return token_list  # List[B, 1, N_tokens, embed_dim]

    @torch.no_grad()
    def sequential_predict(
        self,
        images: T,
        conditioning: T,
        num_steps: int,
        num_samplings: int,
        *args,
        **kwargs,
    ) -> List[T]:
        B = images.size(0)
        tokens_gt = self.tokenize(images)

        masks, masks_pred = self.get_masks_masks_pred_indices(B, num_steps)
        tokens = torch.zeros(
            B, self.generator.seq_len, self.generator.token_embed_dim
        ).to(self.model_cfg.device)

        tokens_pred = []

        for _, mask, mask_to_pred in zip(range(num_steps), masks, masks_pred):
            cur_tokens = tokens.clone()
            class_embedding = self.generator.class_emb(conditioning)

            x = self.generator.forward_mae_encoder(tokens, mask, class_embedding)
            z = self.generator.forward_mae_decoder(x, mask)

            # sample token latents for this step
            z = z[mask_to_pred != 0].view(B, 1, -1, z.shape[2])
            z = z.repeat(
                1, num_samplings, 1, 1
            )  # B, num_samplings, N_tokens, embed_dim
            tokens_pred.append(
                self.generator.diffloss.sample(z.view(-1, z.shape[3]), 1, 1).view(
                    B, num_samplings, -1, self.generator.token_embed_dim
                )
            )

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = tokens_gt[
                mask_to_pred.nonzero(as_tuple=True)
            ]
            tokens = cur_tokens.clone()

        return tokens_pred

    @torch.no_grad()
    def sample(self, z: T, B: int) -> T:
        return self.generator.diffloss.sample(z, 1, 1).view(
            B, -1, self.generator.token_embed_dim
        )  # B, N_tokens, embed_dim

    @torch.no_grad()
    def get_seconds_per_image(self) -> float:
        B = self.model_cfg.batch_size
        times = []
        for _ in range(10):
            start = time()
            sampled_tokens = self.generator.sample_tokens(
                bsz=B,
                labels=torch.randint(0, 1000, (B,), device=self.model_cfg.device),
            )
            sampled_images = self.tokenizer.decode(sampled_tokens / 0.2325)
            times.append(time() - start)

        return sum(times) / (len(times) * B)

    @torch.no_grad()
    def get_alpha_cumprod(self, timestep: int):
        return self.generator.diffloss.train_diffusion.alphas_cumprod[timestep]

    @torch.no_grad()
    def get_noise_prediction_input(self, images: T, conditioning: T, is_cfg: bool):
        B = images.size(0)
        z, target, mask = self.forward(
            images, conditioning, is_cfg, return_target_mask=True
        )
        return z, target, mask

    @torch.no_grad()
    def predict_noise(self, z: T, target: T, mask: T, timestep: int) -> T:
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(
            self.generator.diffusion_batch_mul, 1
        )

        # note the reshape to (batch_size * sequence_length, C)
        z = z.reshape(bsz * seq_len, -1).repeat(self.generator.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz * seq_len).repeat(self.generator.diffusion_batch_mul)
        t = torch.ones(target.shape[0], device="cuda").long() * timestep
        model_kwargs = dict(c=z)

        # noise sample
        noise = torch.randn_like(target)
        x_t = self.generator.diffloss.train_diffusion.q_sample(target, t, noise=noise)
        model_output = self.generator.diffloss.net(x_t, t, **model_kwargs)

        # split into noise prediction and variance prediction
        B, C = x_t.shape[:2]
        assert model_output.shape == (B, C * 2, *x_t.shape[2:])
        model_output, model_var_values = torch.split(model_output, C, dim=1)

        return model_output


class MARProfiler(GeneralVARProfiler):
    @torch.no_grad()
    def forward_train(self, x: T, mask: T, class_embedding: T, gt_latents: T) -> T:
        self.generator: MAR
        x = self.generator.forward_mae_encoder(x, mask, class_embedding)
        z = self.generator.forward_mae_decoder(x, mask)
        loss = self.loss_fn(z, gt_latents, mask)
        return loss

    @torch.no_grad()
    def forward_generate(
        self, step: int, tokens: T, class_embedding: T, orders: T, mask: T
    ) -> Tuple[T, T]:
        return self.forward_single_step(step, tokens, class_embedding, orders, mask)

    @torch.no_grad()
    def forward_single_step(
        self,
        step: int,
        tokens: T,
        class_embedding: T,
        orders: T,
        mask: T,
    ) -> Tuple[T, T]:
        bsz = 1
        num_iter = 64
        temperature = 1.0
        cfg = 1.5
        cur_tokens = tokens.clone()

        tokens = torch.cat([tokens, tokens], dim=0)
        class_embedding = torch.cat(
            [class_embedding, self.generator.fake_latent.repeat(bsz, 1)], dim=0
        )
        mask = torch.cat([mask, mask], dim=0)

        x = self.generator.forward_mae_encoder(tokens, mask, class_embedding)
        z = self.generator.forward_mae_decoder(x, mask)

        mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / num_iter)
        mask_len = torch.Tensor([np.floor(self.generator.seq_len * mask_ratio)]).cuda()

        mask_len = torch.maximum(
            torch.Tensor([1]).cuda(),
            torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len),
        )

        mask_next = mask_by_order(
            mask_len[0],
            orders,
            bsz,
            self.generator.seq_len,
            device=self.model_cfg.device,
        )
        if step >= num_iter - 1:
            mask_to_pred = mask[:bsz].bool()
        else:
            mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
        mask = mask_next
        mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

        z = z[mask_to_pred.nonzero(as_tuple=True)]
        cfg_iter = (
            1
            + (cfg - 1)
            * (self.generator.seq_len - mask_len[0])
            / self.generator.seq_len
        )

        sampled_token_latent = self.generator.diffloss.sample(z, temperature, cfg_iter)
        sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
        mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

        cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
        tokens = cur_tokens.clone()
        return tokens, mask

    @torch.no_grad()
    def loss_fn(self, z: T, target: T, mask: T) -> T:

        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(
            self.generator.diffusion_batch_mul, 1
        )
        z = z.reshape(bsz * seq_len, -1).repeat(self.generator.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz * seq_len).repeat(self.generator.diffusion_batch_mul)

        t = torch.randint(
            0,
            self.generator.diffloss.train_diffusion.num_timesteps,
            (target.shape[0],),
            device=self.model_cfg.device,
        )
        model_kwargs = dict(c=z)
        loss_dict = self.generator.diffloss.train_diffusion.training_losses(
            self.generator.diffloss.net, target, t, model_kwargs
        )
        loss = loss_dict["loss"]
        loss = (loss * mask) / mask.sum()
        return loss
