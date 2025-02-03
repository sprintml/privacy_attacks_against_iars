import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from tqdm import tqdm

from itertools import product

from analysis.evaluate import get_tpr_fpr, get_auc, get_accuracy
from analysis.utils import (
    set_plt,
    MODELS_NAME_MAPPING,
    MODELS_ORDER,
    MODELS,
    RUN_ID,
    MODEL_MIA_INDICES_MAPPING,
    MODEL_MIA_ATTACK_NAME_MAPPING,
    MIAS_CITATIONS,
    MIAS_ORDER,
)

import multiprocessing as mp

from typing import Tuple, List

set_plt()

N_PROC = 20
SIZE = 5_000
PATH_TO_FEATURES = "out/features"
PATH_TO_PLOTS = "analysis/plots/mia_performance"


def get_features() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    all_members, all_nonmembers = [], []
    for model in tqdm(MODELS):
        try:
            members = np.load(
                f"{PATH_TO_FEATURES}/{model}_{MODEL_MIA_ATTACK_NAME_MAPPING[model]}_{RUN_ID}_imagenet_train.npz",
                allow_pickle=True,
            )["data"]
            nonmembers = np.load(
                f"{PATH_TO_FEATURES}/{model}_{MODEL_MIA_ATTACK_NAME_MAPPING[model]}_{RUN_ID}_imagenet_val.npz",
                allow_pickle=True,
            )["data"]
            all_members.append(members)
            all_nonmembers.append(nonmembers)
        except FileNotFoundError:
            print(
                f"Missing {model}, {PATH_TO_FEATURES}/{model}_{MODEL_MIA_ATTACK_NAME_MAPPING[model]}_{RUN_ID}_imagenet"
            )
            all_members.append(None)
            all_nonmembers.append(None)
            continue
    return all_members, all_nonmembers


def get_row(members: np.ndarray, nonmembers: np.ndarray, model: str) -> List:
    out = []
    NREPS = 250
    np.random.seed(42)

    members_lower = True
    for ft_idx, _ in tqdm(
        product(range(members.shape[2]), range(NREPS)),
        desc=model,
        total=members.shape[2] * NREPS,
    ):
        indices = np.random.permutation(len(members))[:SIZE]
        members_feature = members[indices, 0, ft_idx]
        nonmembers_feature = nonmembers[indices, 0, ft_idx]

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
                f"{tpr*100:.2f}",
                f"{auc*100:.2f}",
                f"{accuracy*100:.2f}",
            ]
        )
    pd.DataFrame(
        out,
        columns=["FTIDX", "Model", "TPR@FPR=1\%", "AUC", "Accuracy"],
    ).to_csv(
        f"{PATH_TO_PLOTS}/tmp/mia_performance_{model}.csv",
        index=False,
    )


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
            if f != "mia_performance.csv" and f.endswith(".csv")
        ]
    )
    return data


def get_agg_data(df: pd.DataFrame) -> pd.DataFrame:
    stds = df.groupby(["FTIDX", "Model"]).std()
    means = df.groupby(["FTIDX", "Model"]).mean()
    stds = stds.rename(
        columns={
            "TPR@FPR=1\%": "TPR@FPR=1\%_std",
            "AUC": "AUC_std",
            "Accuracy": "Accuracy_std",
        }
    ).reset_index()
    means = means.rename(
        columns={
            "TPR@FPR=1\%": "TPR@FPR=1\%_mean",
            "AUC": "AUC_mean",
            "Accuracy": "Accuracy_mean",
        }
    ).reset_index()

    data = pd.merge(means, stds, on=["FTIDX", "Model"])
    return data


def get_main_paper_table(data: pd.DataFrame) -> pd.DataFrame:
    dfs = []
    for model in MODELS:
        indices_mapping = MODEL_MIA_INDICES_MAPPING[model]
        for mia, indices in indices_mapping.items():
            tmp_df = data.loc[
                (data["FTIDX"].isin(indices)) & (data["Model"] == model)
            ].copy()
            if len(indices) == 1:
                tmp_df["MIA"] = MIAS_CITATIONS[mia]
                tmp_df["Model"] = tmp_df["Model"].apply(
                    lambda x: MODELS_NAME_MAPPING[x]
                )
                for metric in ["TPR@FPR=1\%", "AUC", "Accuracy"]:
                    tmp_df[metric] = tmp_df[f"{metric}_mean"].apply(
                        lambda x: f"{x:.2f}"
                    ) + tmp_df[f"{metric}_std"].apply(
                        lambda x: f"{{\\tiny $\pm${x:.2f}}}"
                    )
                dfs.append(tmp_df[["Model", "MIA", "TPR@FPR=1\%", "AUC", "Accuracy"]])
                continue
            max_indices = tmp_df.groupby(["Model"])[
                ["TPR@FPR=1\%_mean", "AUC_mean", "Accuracy_mean"]
            ].idxmax()
            tmp_df = tmp_df.loc[max_indices["TPR@FPR=1\%_mean"]]
            tmp_df["MIA"] = MIAS_CITATIONS[mia]
            tmp_df["Model"] = tmp_df["Model"].apply(lambda x: MODELS_NAME_MAPPING[x])
            for metric in ["TPR@FPR=1\%", "AUC", "Accuracy"]:
                tmp_df[metric] = tmp_df[f"{metric}_mean"].apply(
                    lambda x: f"{x:.2f}"
                ) + tmp_df[f"{metric}_std"].apply(lambda x: f"{{\\tiny $\pm${x:.2f}}}")
            dfs.append(tmp_df[["Model", "MIA", "TPR@FPR=1\%", "AUC", "Accuracy"]])

    df = pd.concat(dfs)
    df.to_csv(f"{PATH_TO_PLOTS}/mia_performance_agg.csv", index=False)
    tpr_df = df.pivot(index="MIA", columns="Model", values="TPR@FPR=1\%").loc[
        MIAS_ORDER
    ][MODELS_ORDER]
    tpr_df = tpr_df.fillna("---")
    tpr_df.index.name = None
    tpr_df.columns.name = None
    tpr_df.to_latex(
        f"{PATH_TO_PLOTS}/mia_performance_main.tex",
        escape=False,
        column_format="c" * (len(MODELS) + 1),
    )

    auc_df = df.pivot(index="MIA", columns="Model", values="AUC").loc[MIAS_ORDER][
        MODELS_ORDER
    ]
    auc_df = auc_df.fillna("---")
    auc_df.index.name = None
    auc_df.columns.name = None
    auc_df.to_latex(
        f"{PATH_TO_PLOTS}/mia_performance_auc.tex",
        escape=False,
        column_format="c" * (len(MODELS) + 1),
    )


def main():
    os.makedirs(PATH_TO_PLOTS, exist_ok=True)
    os.makedirs(os.path.join(PATH_TO_PLOTS, "tmp"), exist_ok=True)
    np.random.seed(42)

    data = get_data()
    df = pd.DataFrame(
        data, columns=["FTIDX", "Model", "TPR@FPR=1\%", "AUC", "Accuracy"]
    )
    df.to_csv(f"{PATH_TO_PLOTS}/mia_performance.csv", index=False)
    df = pd.read_csv(f"{PATH_TO_PLOTS}/mia_performance.csv")

    get_main_paper_table(get_agg_data(df))


if __name__ == "__main__":
    main()
