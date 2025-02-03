from src.attacks import FeatureExtractor
from torch import Tensor as T
import torch

from tqdm import tqdm
from itertools import product

import numpy as np

top_tokens = {
    "var_30": list(range(5)),
    "rar_xxl": [0, 1, 5, 14, 30],
    "mar_h": [0, 1, 2, 7, 15],
}

ATTACKS = {
    "var_30": "mem_info",
    "rar_xxl": "mem_info",
    "mar_h": "mem_info_mar",
}

TOP_CLASS_SAMPLES = list(range(5))


class GenerateCandidates(FeatureExtractor):
    def get_data(self, split: str) -> np.ndarray:
        RUN_IDS = [f"1M_{i}" for i in range(8)] if split == "train" else ["50k"]
        out = []
        for run_id in tqdm(RUN_IDS):
            try:
                data = np.load(
                    f"out/features/{self.model_cfg.name}_{ATTACKS[self.model_cfg.name]}_{run_id}_imagenet_{split}.npz",
                    allow_pickle=True,
                )
                out.append(data["data"])
            except FileNotFoundError:
                print(
                    f"File not found: {self.model_cfg.name}_{ATTACKS[self.model_cfg.name]}_{run_id}_imagenet_{split}.npz"
                )
        return np.concatenate(out, axis=0)

    def get_batches(self, lst: list, batch_size: int) -> list:
        return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]

    @torch.no_grad()
    def run(self, *args, **kwargs) -> T:
        TOP_TOKENS = top_tokens[self.model_cfg.name]
        members_features = self.get_data(self.dataset_cfg.split)
        members_features = torch.from_numpy(members_features)
        print("Data loaded")

        torch.manual_seed(0)
        classes = torch.randperm(1000)[: self.attack_cfg.nclasses][
            self.attack_cfg.gsplit
            * (self.attack_cfg.nclasses // self.attack_cfg.n_gsplits) : (
                self.attack_cfg.gsplit + 1
            )
            * (self.attack_cfg.nclasses // self.attack_cfg.n_gsplits)
        ]
        scores: T = self.model.get_memorization_scores(members_features, 1)
        sample_classes = members_features[:, 0, -1].clone()
        ins = []

        for cls, top_k in tqdm(
            product(classes, TOP_CLASS_SAMPLES), desc="Getting Samples"
        ):
            target_tokens, label_B, s_idx = self.model.get_target_label_memorization(
                members_features, scores, sample_classes, cls, top_k
            )
            ins.append((target_tokens, label_B, s_idx))
        batches = self.get_batches(ins, self.model_cfg.batch_size)

        out = []
        for batch in tqdm(batches, desc="Generating Samples"):
            target_tokens = torch.cat([x[0] for x in batch], dim=0)
            label_B = torch.cat([x[1] for x in batch], dim=0)
            s_idx = torch.cat([x[2] for x in batch], dim=0)

            pred = []
            for top_tokens in TOP_TOKENS:
                pred_tokens = self.model.generate_single_memorization(
                    top_tokens,
                    self.model.tokens_to_token_list(target_tokens),
                    label_B,
                    self.attack_cfg.std,
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

        out = torch.cat(out, dim=0).cpu().numpy()
        np.savez(
            f"{self.config.path_to_features}/{self.model_cfg.name}_{ATTACKS[self.model_cfg.name]}_memorized_imagenet_{self.dataset_cfg.split}_{self.attack_cfg.std}.npz",
            data=out,
        )
