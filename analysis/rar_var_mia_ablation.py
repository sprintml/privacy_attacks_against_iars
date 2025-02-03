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
    IARs,
    RUN_ID,
)

from typing import Tuple, List

MODELS = [model for model in IARs if model not in ["mar_b", "mar_l", "mar_h"]]
ATTACK_NAME_MAPPING = {
    "llm_mia": "Naive",
    "llm_mia_cfg": "After changes",
}

set_plt()

SIZE = 5_000
PATH_TO_FEATURES = "out/features"
PATH_TO_PLOTS = "analysis/plots/rar_var_mia_ablation"


def get_data() -> List:
    out = []
    for model, attack in tqdm(product(MODELS, ATTACK_NAME_MAPPING.keys())):
        np.random.seed(42)
        try:
            members = np.load(
                f"{PATH_TO_FEATURES}/{model}_{attack}_{RUN_ID}_imagenet_train.npz",
                allow_pickle=True,
            )["data"]
            nonmembers = np.load(
                f"{PATH_TO_FEATURES}/{model}_{attack}_{RUN_ID}_imagenet_val.npz",
                allow_pickle=True,
            )["data"]
        except FileNotFoundError:
            print(f"Missing {model}, {PATH_TO_FEATURES}/{model}_{attack}_{RUN_ID}")
            continue
        members_lower = True
        for ft_idx, _ in product(range(members.shape[2]), range(10)):
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
            auc = get_auc(
                members_feature, nonmembers_feature, members_lower=members_lower
            )
            accuracy = get_accuracy(
                members_feature, nonmembers_feature, members_lower=members_lower
            )

            out.append(
                [
                    ft_idx,
                    MODELS_NAME_MAPPING[model],
                    attack,
                    f"{tpr*100:.2f}",
                    f"{auc*100:.2f}",
                    f"{accuracy*100:.2f}",
                ]
            )

    return out


def get_ablation_table(data: pd.DataFrame) -> pd.DataFrame:
    data = (
        data.groupby(["Model", "FTIDX", "Attack"])["TPR@FPR=1\%"].mean().reset_index()
    )
    data = data.groupby(["Model", "Attack"])["TPR@FPR=1\%"].max().reset_index()
    data = data.pivot(index="Attack", columns="Model", values="TPR@FPR=1\%")[
        [model for model in MODELS_ORDER if model in data["Model"].unique()]
    ]
    data = pd.concat(
        [
            data.loc[data.index == "llm_mia"],
            data.loc[data.index == "llm_mia_cfg"] - data.loc["llm_mia"],
        ]
    )
    data.loc[data.index == "llm_mia"] = data.loc[data.index == "llm_mia"].map(
        lambda x: f"{x:.3f}"
    )
    data.loc[data.index != "llm_mia"] = data.loc[data.index != "llm_mia"].map(
        lambda x: f"\\textbf{{{'+' if x>0 else''}{x:.2f}}}"
    )
    data.index = data.index.map(ATTACK_NAME_MAPPING)
    data.to_latex(f"{PATH_TO_PLOTS}/var_rar_ablation.tex", escape=False)


def main():
    os.makedirs(PATH_TO_PLOTS, exist_ok=True)
    np.random.seed(42)

    data = get_data()
    df = pd.DataFrame(
        data, columns=["FTIDX", "Model", "Attack", "TPR@FPR=1\%", "AUC", "Accuracy"]
    )
    df.to_csv(f"{PATH_TO_PLOTS}/mia_performance.csv", index=False)
    df = pd.read_csv(f"{PATH_TO_PLOTS}/mia_performance.csv")

    get_ablation_table(df)

if __name__ == "__main__":
    main()