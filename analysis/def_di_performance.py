import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from tqdm import tqdm

import multiprocessing as mp
from itertools import product
from sklearn.preprocessing import MinMaxScaler

from analysis.evaluate import get_p_value
from analysis.utils import (
    set_plt,
    IARs,
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
PATH_TO_PLOTS = "analysis/plots/def_di_performance"

stds = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]


def preprocess_features(
    members: np.ndarray, nonmembers: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    B = members.shape[0]

    members[np.isnan(members)] = 0
    nonmembers[np.isnan(nonmembers)] = 0

    scaler = MinMaxScaler()
    data = np.concat([members.reshape(B, -1), nonmembers.reshape(B, -1)], axis=0)
    data = scaler.fit_transform(data)

    members = data[: members.shape[0]]
    nonmembers = data[members.shape[0] :]
    return members.sum(axis=1), nonmembers.sum(axis=1)


def get_row(
    model: str,
    n: int,
    members: np.ndarray,
    nonmembers: np.ndarray,
):
    out = []
    NREPS = 1000
    np.random.seed(42)
    members_lower = True
    for (idx, std), _ in tqdm(
        product(enumerate(stds), range(NREPS)),
        desc=f"{model=}, {n=}",
        total=len(stds) * NREPS * members.shape[2],
    ):
        indices = np.random.choice(10_000, n, replace=False)
        try:
            members_feature = members[indices, idx]
        except IndexError:
            continue
        nonmembers_feature = nonmembers[indices, idx]

        members_eval, nonmembers_eval = (
            members_feature.copy(),
            nonmembers_feature.copy(),
        )
        members_scores, nonmembers_scores = preprocess_features(
            members_eval, nonmembers_eval
        )
        pvalue, _ = get_p_value(
            members_scores, nonmembers_scores, members_lower=members_lower
        )
        out.append(
            [
                model,
                std,
                n,
                pvalue,
            ]
        )
    pd.DataFrame(
        out,
        columns=["Model", "STD", "n", "pvalue"],
    ).to_csv(
        f"{PATH_TO_PLOTS}/tmp/pvalue_per_sample_{model}_{n}.csv",
        index=False,
    )


def remove_features(features, SURP_IDX):
    filtered_features = features[
        :, :, np.setdiff1d(np.arange(features.shape[2]), SURP_IDX)
    ]
    return filtered_features


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

            if model in ["mar_l", "mar_b", "mar_h"]:
                members = remove_features(members, np.arange(7, 10))
                nonmembers = remove_features(nonmembers, np.arange(7, 10))

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
    nsamples = np.array(
        [n for n in range(2, 11, 1)]
        + [n for n in range(20, 110, 10)]
        + [n for n in range(200, 1100, 100)]
        + [n for n in range(2000, 11_000, 1000)]
    )
    with mp.Pool(N_PROC) as pool:
        for idx, model in enumerate(MODELS):
            m, nm = members[idx], nonmembers[idx]
            if m is None:
                continue
            for n in nsamples:
                pool.apply_async(
                    get_row,
                    args=(model, n, m, nm),
                )

        pool.close()
        pool.join()

    data = pd.concat(
        [
            pd.read_csv(os.path.join(PATH_TO_PLOTS, "tmp", f))
            for f in tqdm(os.listdir(os.path.join(PATH_TO_PLOTS, "tmp")))
            if f != "pvalue_per_sample.csv" and f.endswith(".csv")
        ]
    )
    return data


def get_agg_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.groupby(["Model", "STD", "n"]).pvalue.mean().reset_index()
    data = data.loc[data.pvalue <= 0.01].groupby(["Model", "STD"]).n.min().reset_index()
    return data


def main():
    os.makedirs(os.path.join(PATH_TO_PLOTS, "tmp"), exist_ok=True)
    np.random.seed(42)

    data = get_data()
    df = pd.DataFrame(
        data,
        columns=["Model", "STD", "n", "pvalue"],
    )
    df.to_csv(f"{PATH_TO_PLOTS}/pvalue_per_sample.csv", index=False)
    df = pd.read_csv(f"{PATH_TO_PLOTS}/pvalue_per_sample.csv")

    get_agg_data(df).to_csv(f"{PATH_TO_PLOTS}/pvalue_per_sample_agg.csv", index=False)


if __name__ == "__main__":
    main()
