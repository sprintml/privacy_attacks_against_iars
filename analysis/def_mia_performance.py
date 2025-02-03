import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from tqdm import tqdm

import multiprocessing as mp
from itertools import product

from analysis.evaluate import get_tpr_fpr, get_auc, get_accuracy
from analysis.utils import (
    set_plt,
    IARs,
    MODELS,
    RUN_ID,
)

from typing import Tuple, List

set_plt()

RUN_ID = "10k"
MODELS = IARs
N_PROC = 20

ATTACK = {
    "mar_b": "defense_loss",
    "mar_l": "defense_loss",
    "mar_h": "defense_loss",
}

SIZE = 5_000
PATH_TO_FEATURES = "out/features"
PATH_TO_PLOTS = "analysis/plots/def_mia_performance"

stds = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]


def get_row(members: np.ndarray, nonmembers: np.ndarray, model: str):
    out = []
    NREPS = 250
    np.random.seed(42)
    members_lower = True
    for ft_idx, (idx, std), _ in tqdm(
        product(range(members.shape[2]), enumerate(stds), range(NREPS)),
        desc=model,
        total=len(stds) * NREPS * members.shape[2],
    ):
        indices = np.random.permutation(len(members))[:SIZE]
        try:
            members_feature = members[indices, idx, ft_idx]
        except IndexError:
            continue
        nonmembers_feature = nonmembers[indices, idx, ft_idx]

        members_feature[np.isnan(members_feature)] = 0
        nonmembers_feature[np.isnan(nonmembers_feature)] = 0

        tpr = get_tpr_fpr(
            members_feature,
            nonmembers_feature,
            fpr_threshold=0.01,
            members_lower=members_lower,
        )
        auc = get_auc(members_feature, nonmembers_feature, members_lower=members_lower)
        accuracy = get_accuracy(
            members_feature, nonmembers_feature, members_lower=members_lower
        )

        out.append(
            [
                ft_idx,
                model,
                std,
                f"{tpr*100:.2f}",
                f"{auc*100:.2f}",
                f"{accuracy*100:.2f}",
            ]
        )
    pd.DataFrame(
        out,
        columns=["FTIDX", "Model", "STD", "TPR@FPR=1\%", "AUC", "Accuracy"],
    ).to_csv(
        f"{PATH_TO_PLOTS}/tmp/def_mia_performance_{model}.csv",
        index=False,
    )


def get_features() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    all_members, all_nonmembers = [], []
    for model in tqdm(MODELS):
        try:
            attack = ATTACK.get(model, "defense")
            members = np.load(
                f"{PATH_TO_FEATURES}/{model}_{attack}_{RUN_ID}_imagenet_train.npz",
                allow_pickle=True,
            )["data"]
            nonmembers = np.load(
                f"{PATH_TO_FEATURES}/{model}_{attack}_{RUN_ID}_imagenet_val.npz",
                allow_pickle=True,
            )["data"]
            all_members.append(members)
            all_nonmembers.append(nonmembers)
        except FileNotFoundError:
            print(
                f"Missing {model}, {PATH_TO_FEATURES}/{model}_{attack}_{RUN_ID}_imagenet"
            )
            all_members.append(None)
            all_nonmembers.append(None)
            continue
    return all_members, all_nonmembers


def get_data() -> list:
    members, nonmembers = get_features()
    print("Features loaded")

    with mp.Pool(N_PROC) as pool:
        for idx, model in enumerate(MODELS):
            m, nm = members[idx], nonmembers[idx]
            if m is None:
                continue
            pool.apply_async(
                get_row,
                args=(m, nm, model),
            )

        pool.close()
        pool.join()

    data = pd.concat(
        [
            pd.read_csv(os.path.join(PATH_TO_PLOTS, "tmp", f))
            for f in tqdm(os.listdir(os.path.join(PATH_TO_PLOTS, "tmp")))
            if f != "def_mia_performance.csv" and f.endswith(".csv")
        ]
    )
    return data


def get_agg_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.groupby(["FTIDX", "Model", "STD"])["TPR@FPR=1\%"].mean().reset_index()
    data = data.groupby(["Model", "STD"])["TPR@FPR=1\%"].max().reset_index()
    return data


def main():
    os.makedirs(os.path.join(PATH_TO_PLOTS, "tmp"), exist_ok=True)
    np.random.seed(42)

    data = get_data()
    df = pd.DataFrame(
        data,
        columns=["FTIDX", "Model", "STD", "TPR@FPR=1\%", "AUC", "Accuracy"],
    )
    df.to_csv(f"{PATH_TO_PLOTS}/def_mia_performance.csv", index=False)
    df = pd.read_csv(f"{PATH_TO_PLOTS}/def_mia_performance.csv")

    get_agg_data(df).to_csv(f"{PATH_TO_PLOTS}/def_mia_performance_agg.csv", index=False)

if __name__ == "__main__":
    main()
