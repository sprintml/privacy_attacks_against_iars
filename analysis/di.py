import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


from itertools import product
import multiprocessing as mp
from sklearn.preprocessing import MinMaxScaler


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from analysis.evaluate import get_p_value
from analysis.utils import (
    set_plt,
    ATTACKS_NAME_MAPPING,
    MODELS_NAME_MAPPING,
    MODELS_ORDER,
    RESAMPLING_CNT,
    ATTACKS,
    MODELS,
    RUN_ID,
    IARs,
)
import pandas as pd

from typing import Tuple

set_plt()

N_PROC = 80

MODELS = IARs
ATTACKS = ["llm_mia_loss", "llm_mia_cfg"]


PVALUES = [0.01, 0.05]
NSAMPLES_TO_SHOW = [10, 100, 500, 1000, 5000, 10000]
REMOVE_SURP = True
REMOVE_CAMIA = True

PATH_TO_FEATURES = "out/features"
PATH_TO_PLOTS = "analysis/plots/di"

CAMIA_IDX_MAR = np.arange(7, 10)


def remove_features(features, SURP_IDX):
    filtered_features = features[
        :, :, np.setdiff1d(np.arange(features.shape[2]), SURP_IDX)
    ]
    return filtered_features


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


def get_scores():
    all_members, all_nonmembers = [], []
    for attack, model in product(ATTACKS, MODELS):
        try:
            members = np.load(
                f"{PATH_TO_FEATURES}/{model}_{attack}_{RUN_ID}_imagenet_train.npz",
                allow_pickle=True,
            )
            nonmembers = np.load(
                f"{PATH_TO_FEATURES}/{model}_{attack}_{RUN_ID}_imagenet_val.npz",
                allow_pickle=True,
            )
        except FileNotFoundError:
            all_members.append([None, None])
            all_nonmembers.append([None, None])
            continue
        members_lower = members["metadata"][()]["members_lower"]

        members = members["data"]
        nonmembers = nonmembers["data"]

        if (
            model in ["mar_l", "mar_b", "mar_h"]
            and attack in ["di", "llm_mia_loss"]
            and REMOVE_CAMIA
        ):
            members = remove_features(members, CAMIA_IDX_MAR)
            nonmembers = remove_features(nonmembers, CAMIA_IDX_MAR)
        all_members.append([members, members_lower])
        all_nonmembers.append([nonmembers, members_lower])

    return all_members, all_nonmembers


def get_row(
    model: str,
    attack: str,
    n: int,
    indices_total: np.ndarray,
    members: np.ndarray,
    nonmembers: np.ndarray,
    members_lower: bool,
):
    out = []
    for r in tqdm(
        range(RESAMPLING_CNT),
        desc=f"{model} {attack} {n=}",
        total=RESAMPLING_CNT,
    ):
        try:
            indices = np.random.choice(indices_total, n, replace=False)
            members_eval, nonmembers_eval = (
                members[indices].copy(),
                nonmembers[indices].copy(),
            )
            members_scores, nonmembers_scores = preprocess_features(
                members_eval, nonmembers_eval
            )
            pvalue, is_correct_order = get_p_value(
                members_scores, nonmembers_scores, members_lower=members_lower
            )
            out.append(
                [
                    ATTACKS_NAME_MAPPING[attack],
                    MODELS_NAME_MAPPING[model],
                    n,
                    pvalue,
                    is_correct_order,
                    r,
                ]
            )
        except Exception as e:
            print("Exception", model, attack, n, r, e)
            exit()
    pd.DataFrame(
        out,
        columns=["Attack", "Model", "n", "pvalue", "is_correct_order", "r"],
    ).to_csv(
        f"{PATH_TO_PLOTS}/tmp/pvalue_per_sample_{model}_{attack}_{n}.csv",
        index=False,
    )
    print(f"Done {model} {attack}, {n=}")


def get_data() -> list:
    nsamples = np.array(
        [n for n in range(2, 11, 1)]
        + [n for n in range(20, 110, 10)]
        + [n for n in range(200, 1100, 100)]
        + [n for n in range(2000, 11_000, 1000)]
    )
    indices_total = np.arange(10_000)

    members, nonmembers = get_scores()
    print("Scores loaded")

    with mp.Pool(N_PROC) as pool:
        for idx, (attack, model) in enumerate(product(ATTACKS, MODELS)):
            m, nm, ml = members[idx][0], nonmembers[idx][0], members[idx][1]
            if m is None:
                continue
            for n in nsamples:
                pool.apply_async(
                    get_row,
                    args=(model, attack, n, indices_total, m, nm, ml),
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
    data = data.sort_values(
        by="Model", key=lambda col: col.map(lambda e: MODELS_ORDER.index(e))
    )
    return data


def main():
    os.makedirs(os.path.join(PATH_TO_PLOTS, "tmp"), exist_ok=True)
    np.random.seed(42)
    try:
        df = pd.read_csv(f"{PATH_TO_PLOTS}/pvalue_per_sample.csv")
    except FileNotFoundError:
        data = get_data()
        df = pd.DataFrame(
            data,
            columns=["Attack", "Model", "n", "pvalue", "is_correct_order", "r"],
        )
        df.to_csv(f"{PATH_TO_PLOTS}/pvalue_per_sample.csv", index=False)
    df = df.groupby(["Model", "n"]).pvalue.mean().reset_index()
    df = df.loc[df.pvalue <= 0.01].groupby("Model").n.min()
    df.to_csv(f"{PATH_TO_PLOTS}/di_results.csv")


if __name__ == "__main__":
    main()
