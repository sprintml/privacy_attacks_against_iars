from omegaconf import open_dict
from hydra import initialize, compose
import sys
import torch
import os

import numpy as np
from torch import Tensor as T

sys.path.append("./VAR")
sys.path.append("./mar")
sys.path.append("./rar")

from src import gen_models
from src.models import GeneralVARWrapper, MARWrapper
from VAR.models.var import VAR
from rar.modeling.rar import RAR
from mar.models.mar import MAR
from VAR.models.vqvae import VQVAE

from tqdm import tqdm
from itertools import product
from typing import List, Tuple


import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mar_h")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--gsplit", type=int, default=0)
    parser.add_argument("--std", type=float, default=0.0)
    args = parser.parse_args()
    return args


args = parse_args()

device = "cuda"
MODEL = args.model
SPLIT = args.split

top_tokens = {
    "var_16": list(range(5)),
    "var_20": list(range(5)),
    "var_24": list(range(5)),
    "var_30": list(range(5)),
    "rar_xxl": [0, 1, 5, 14, 30],
    "mar_b": list(range(5)),
    "mar_l": list(range(5)),
    "mar_h": [0, 1, 2, 7, 15],
}

ATTACK = {
    "var_16": "mem_info",
    "var_20": "mem_info",
    "var_24": "mem_info",
    "var_30": "mem_info",
    "rar_xxl": "mem_info",
    "mar_b": "mem_info_mar",
    "mar_l": "mem_info_mar",
    "mar_h": "mem_info_mar",
}[MODEL]

BATCH_SIZE = 64
NSPLITS = 1

NCLASSES = 1000  # 100
TOP_TOKENS = top_tokens[MODEL]
TOP_CLASS_SAMPLES = list(range(5))

RUN_IDS = [f"1M_{i}" for i in range(8)] if SPLIT == "train" else ["50k"]


def get_data(split: str) -> np.ndarray:
    out = []
    for run_id in tqdm(RUN_IDS):
        try:
            data = np.load(
                f"out/features/{MODEL}_{ATTACK}_{run_id}_imagenet_{split}.npz",
                allow_pickle=True,
            )
            out.append(data["data"])
        except FileNotFoundError:
            print(f"File not found: {MODEL}_{ATTACK}_{run_id}_imagenet_{split}.npz")
    return np.concatenate(out, axis=0)


def get_single_scores(features: T, agg: str) -> T:
    if agg == "mean":
        return features.mean(dim=1)
    if agg == "max":
        return features.max(dim=1)
    if agg == "min":
        return features.min(dim=1)
    if agg == "sum":
        return features.sum(dim=1)


def get_scores_mar(data: T, _, __) -> T:
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
    distances = data[..., tokens_order] - data[..., 256:-1]
    distances = torch.pow(distances, 2).sum(dim=1)
    return -distances.mean(dim=1)


def get_patchwise_scores(features: T, ft_idx: int, agg: str) -> T:
    out = []
    idx = 0
    patches = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    for patch in patches:
        out.append(get_single_scores(features[:, ft_idx, idx : idx + patch**2], agg))
        idx += patch**2
    return torch.stack(out, dim=1)


def get_model() -> GeneralVARWrapper:
    with initialize(config_path="conf"):
        config = compose("config").cfg
        model_cfg = compose(f"model/{MODEL}").model
        dataset_cfg = compose("dataset/imagenet").dataset

    with open_dict(dataset_cfg):
        dataset_cfg.split = "train"
        dataset_cfg.gpus_cnt = 8
        dataset_cfg.idx = 5

    with open_dict(model_cfg):
        model_cfg.device = device
        model_cfg.seed = config.seed

    return gen_models[model_cfg.name](config, model_cfg, dataset_cfg)


def tokens_to_token_list_var(tokens: T) -> List[T]:
    token_list = []
    idx = 0

    patches = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    ps = np.array([patch**2 for patch in patches])
    for patch in ps:
        scale_target = tokens[:, idx : idx + patch] if idx else tokens[:, [0]]
        token_list.append(scale_target)
        idx += patch
    return token_list


def generate_single_var(
    model: GeneralVARWrapper,
    top: int,
    target_token_list: List[T],
    label_B: T,
    std: float,
) -> T:
    generator: VAR = model.generator
    B = label_B.shape[0]
    total_tokens_used = 0
    for b in generator.blocks:
        b.attn.kv_caching(True)

    sos = cond_BD = generator.class_emb(label_B)
    lvl_pos = generator.lvl_embed(generator.lvl_1L) + generator.pos_1LC
    zero_token_map: T = (
        sos.unsqueeze(1).expand(B, generator.first_l, -1)
        + generator.pos_start.expand(B, generator.first_l, -1)
        + lvl_pos[:, : generator.first_l]
    )

    next_token_map = zero_token_map.clone()

    cur_L = 0
    f_hat = sos.new_zeros(
        B, generator.Cvae, generator.patch_nums[-1], generator.patch_nums[-1]
    )

    pred_tokens = []

    for si, pn in enumerate(generator.patch_nums):  # si: i-th segment
        scale_target = target_token_list[si]
        cur_L += pn * pn
        cond_BD_or_gss = generator.shared_ada_lin(cond_BD)
        x = next_token_map
        for b in generator.blocks:
            x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
        logits_BlV: T = generator.get_logits(x, cond_BD)

        if si < top:
            idx_Bl = scale_target.clone()
            total_tokens_used += len(target_token_list[si][0])
        else:
            logits_BlV = logits_BlV + torch.randn_like(logits_BlV) * std
            idx_Bl = logits_BlV.argmax(dim=2)
        pred_tokens.append(idx_Bl)

        h_BChw = generator.vae_quant_proxy[0].embedding(idx_Bl)
        h_BChw = h_BChw.transpose_(1, 2).reshape(B, generator.Cvae, pn, pn)
        f_hat, next_token_map = generator.vae_quant_proxy[
            0
        ].get_next_autoregressive_input(si, len(generator.patch_nums), f_hat, h_BChw)
        if si != generator.num_stages_minus_1:  # prepare for next stage
            next_token_map = next_token_map.view(B, generator.Cvae, -1).transpose(1, 2)
            next_token_map = (
                generator.word_embed(next_token_map)
                + lvl_pos[:, cur_L : cur_L + generator.patch_nums[si + 1] ** 2]
            )
    pred_tokens = torch.cat(pred_tokens, dim=1)
    return pred_tokens


def generate_single_rar(
    model: GeneralVARWrapper,
    top: int,
    target_token_list: List[T],
    label_B: T,
    std: float,
) -> T:
    generator: RAR = model.generator
    B = label_B.shape[0]
    condition = generator.preprocess_condition(label_B, cond_drop_prob=0.0)
    ids = torch.full((B, 0), -1, device=device)

    generator.enable_kv_cache()

    orders = None

    for step in range(generator.image_seq_len):
        logits = generator.forward_fn(ids, condition, orders=orders, is_sampling=True)
        logits = logits[:, -1]

        if step < top:
            sampled = target_token_list[step]
        else:
            logits = logits + torch.randn_like(logits) * std
            sampled = logits.argmax(dim=1).unsqueeze(1)

        ids = torch.cat((ids, sampled), dim=-1)

    generator.disable_kv_cache()
    return ids


def generate_single_mar(
    model: MARWrapper, top: int, target_token_list: T, label_B: T
) -> T:
    B = label_B.size(0)
    num_steps = 64
    masks, masks_pred = model.get_masks_masks_pred_indices(B, num_steps)
    tokens = torch.zeros(
        B, model.generator.seq_len, model.generator.token_embed_dim
    ).to(model.model_cfg.device)
    class_embedding = model.generator.class_emb(label_B)

    target = target_token_list.clone().view(B, model.generator.seq_len, -1)

    for step, mask, mask_to_pred in zip(range(num_steps), masks, masks_pred):
        cur_tokens = tokens.clone()

        if step < top:
            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = target[
                mask_to_pred.nonzero(as_tuple=True)
            ]
        else:
            x = model.generator.forward_mae_encoder(cur_tokens, mask, class_embedding)
            z = model.generator.forward_mae_decoder(x, mask)
            z = z[mask_to_pred.nonzero(as_tuple=True)]

            tokens_pred = model.generator.diffloss.sample(z, 1, 1)
            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = tokens_pred

        tokens = cur_tokens.clone()

    return tokens.view(B, -1)  # B, seq_len * token_embed_dim


def get_target_label_var(
    members_features: T, scores: T, sample_classes: T, cls: int, k: int
):
    mask = sample_classes == cls
    scores_cls = scores.clone()
    scores_cls[~mask] = -torch.inf
    mem_samples_indices = torch.topk(scores_cls[:, -1], 10).indices
    mem_sample_idx = mem_samples_indices[k]
    label_B = members_features[mem_sample_idx, [0]][0, [-1]].to(device).long()

    target_tokens = members_features[mem_sample_idx, [0]][[0], :680].to(device).long()

    return target_tokens, label_B, mem_sample_idx.unsqueeze(0).to(device)


def get_target_label_rar(
    members_features: T, scores: T, sample_classes: T, cls: int, k: int
):
    mask = sample_classes == cls
    scores_cls = scores.clone()
    scores_cls[~mask] = -torch.inf
    mem_samples_indices = torch.topk(scores_cls, 10).indices
    mem_sample_idx = mem_samples_indices[k]
    label_B = members_features[mem_sample_idx, [0]][0, [-1]].to(device).long()

    target_tokens = members_features[mem_sample_idx, [0]][[0], :256].to(device).long()

    return target_tokens, label_B, mem_sample_idx.unsqueeze(0).to(device)


def get_target_label_mar(
    members_features: T, scores: T, sample_classes: T, cls: int, k: int
):
    mask = sample_classes == cls
    scores_cls = scores.clone()
    scores_cls[~mask] = -torch.inf
    mem_samples_indices = torch.topk(scores_cls, 10).indices
    mem_sample_idx = mem_samples_indices[k]
    label_B = members_features[mem_sample_idx, :][0, [-1]].to(device).long()

    target_tokens = (
        members_features[mem_sample_idx, :][:, :256]
        .permute(1, 0)
        .reshape(1, -1)
        .to(device)
    )

    return target_tokens, label_B, mem_sample_idx.unsqueeze(0).to(device)


def get_batches(lst: list, batch_size: int) -> list:
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


get_scores = {
    "var_16": get_patchwise_scores,
    "var_20": get_patchwise_scores,
    "var_24": get_patchwise_scores,
    "var_30": get_patchwise_scores,
    "rar_xxl": lambda members_features, idx, agg: members_features[
        :, idx, -100:-1
    ].mean(dim=1),
    "mar_b": get_scores_mar,
    "mar_l": get_scores_mar,
    "mar_h": get_scores_mar,
}

get_target_label = {
    "var_16": get_target_label_var,
    "var_20": get_target_label_var,
    "var_24": get_target_label_var,
    "var_30": get_target_label_var,
    "rar_xxl": get_target_label_rar,
    "mar_b": get_target_label_mar,
    "mar_l": get_target_label_mar,
    "mar_h": get_target_label_mar,
}

generate_single = {
    "var_16": generate_single_var,
    "var_20": generate_single_var,
    "var_24": generate_single_var,
    "var_30": generate_single_var,
    "rar_xxl": generate_single_rar,
    "mar_b": generate_single_mar,
    "mar_l": generate_single_mar,
    "mar_h": generate_single_mar,
}

tokens_to_token_list = {
    "var_16": tokens_to_token_list_var,
    "var_20": tokens_to_token_list_var,
    "var_24": tokens_to_token_list_var,
    "var_30": tokens_to_token_list_var,
    "rar_xxl": lambda x: [x[:, [idx]] for idx in range(x.shape[1])],
    "mar_b": lambda x: x,
    "mar_l": lambda x: x,
    "mar_h": lambda x: x,
}


@torch.no_grad()
def run(members_features: T, model: GeneralVARWrapper, gsplit: int, std: float) -> T:
    torch.manual_seed(0)
    classes = torch.randperm(1000)[:NCLASSES][
        gsplit * (NCLASSES // NSPLITS) : (gsplit + 1) * (NCLASSES // NSPLITS)
    ]
    scores: T = get_scores[MODEL](members_features, 1, "mean")
    sample_classes = members_features[:, 0, -1].clone()
    ins = []
    for cls, top_k in tqdm(product(classes, TOP_CLASS_SAMPLES), desc="Getting Samples"):
        target_tokens, label_B, s_idx = get_target_label[MODEL](
            members_features, scores, sample_classes, cls, top_k
        )
        ins.append((target_tokens, label_B, s_idx))
    batches = get_batches(ins, BATCH_SIZE)

    out = []
    for batch in tqdm(batches, desc="Generating Samples"):
        target_tokens = torch.cat([x[0] for x in batch], dim=0)
        label_B = torch.cat([x[1] for x in batch], dim=0)
        s_idx = torch.cat([x[2] for x in batch], dim=0)

        pred = []
        for top_tokens in TOP_TOKENS:
            pred_tokens = generate_single[MODEL](
                model,
                top_tokens,
                tokens_to_token_list[MODEL](target_tokens),
                label_B,
                std,
            )
            pred_tokens = torch.cat(
                [pred_tokens, label_B.unsqueeze(1), s_idx.unsqueeze(1)], dim=1
            )
            pred.append(pred_tokens)

        pred = torch.stack(pred, dim=1)
        target = torch.cat(
            [target_tokens, label_B.unsqueeze(1), s_idx.unsqueeze(1)], dim=1
        ).unsqueeze(1)
        out.append(torch.cat([pred, target], dim=1).cpu())

    return torch.cat(out, dim=0)


def main():
    print(args)
    members_features = get_data(SPLIT)
    members_features = torch.from_numpy(members_features)
    print("Data loaded")

    model = get_model()
    print("Model loaded")

    out = run(members_features, model, args.gsplit, args.std)
    out = out.cpu().numpy()
    np.savez(
        f"out/features/{args.gsplit}_{MODEL}_{ATTACK}_memorized_imagenet_{SPLIT}_{args.std}.npz",
        data=out,
    )


if __name__ == "__main__":
    main()
