import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from tqdm import tqdm

from itertools import product

from analysis.utils import (
    set_plt,
    MODELS_NAME_MAPPING,
    MODELS_ORDER,
    MODELS,
)

from typing import Tuple, List

MODELS = [
    MODELS_NAME_MAPPING[model]
    for model in [
        "var_16",
        "var_20",
        "var_24",
        "var_30",
        "rar_b",
        "rar_l",
        "rar_xl",
        "rar_xxl",
    ]
]
RUN_ID = "10k"  # s = ["10k_mar_default", "10k_mar_better_mask", "10k"]

ATTACKS = ["DI", "LLM_MIA_CFG"]
ATTACK_NAME_MAPPING = {
    "DI_1": "Baseline",
    "DI_0": "$-$ Classifier",
    "LLM_MIA_CFG": "Upgraded MIAs",
}

set_plt()

SIZE = 5_000
PATH_TO_FEATURES = "out/features"
PATH_TO_DI_RESULTS = "analysis/plots/pvalue_per_sample/tmp"
PATH_TO_PLOTS = "analysis/plots/di_improvements_ablation"


def get_row(df: pd.DataFrame) -> Tuple[str, int]:
    model = df.Model.unique()[0]
    df = df.groupby("n").pvalue.mean().reset_index()
    n = df.loc[df.pvalue <= 0.01].n.min()  # .values[0]

    return model, n


def get_data() -> List:
    data = pd.concat(
        [
            pd.read_csv(os.path.join(PATH_TO_DI_RESULTS, f))
            for f in tqdm(os.listdir(PATH_TO_DI_RESULTS))
            if f != "pvalue_per_sample.csv" and f.endswith(".csv")
        ]
    )
    out = []
    for model, attack in tqdm(product(MODELS, ATTACKS)):
        tmp_data = data[(data["Model"] == model) & (data["Attack"] == attack)].copy()
        if attack == "DI":
            for clf in [True, False]:
                tmp_df = tmp_data[tmp_data.CLF == clf].copy()
                out.append([f"DI_{int(clf)}", *get_row(tmp_df)])
        else:
            out.append([attack, *get_row(tmp_data)])

    return out


def get_ablation_table(data: pd.DataFrame) -> pd.DataFrame:
    data = data.pivot(index="Attack", columns="Model", values="N")[
        [model for model in MODELS_ORDER if model in data["Model"].unique()]
    ]
    attacks = [attack for attack in ATTACK_NAME_MAPPING.keys()]
    data = pd.concat(
        [
            data.loc[data.index == attacks[0]],
            *[
                data.loc[data.index == attack] - data.loc[attack_prev]
                for attack_prev, attack in zip(attacks[:-1], attacks[1:])
            ],
        ]
    )
    data.index = data.index.map(ATTACK_NAME_MAPPING)
    data.to_latex(f"{PATH_TO_PLOTS}/di_ablation.tex", escape=False)


def main():
    os.makedirs(PATH_TO_PLOTS, exist_ok=True)
    np.random.seed(42)

    data = get_data()
    df = pd.DataFrame(data, columns=["Attack", "Model", "N"])
    df.to_csv(f"{PATH_TO_PLOTS}/di_ablation.csv", index=False)
    df = pd.read_csv(f"{PATH_TO_PLOTS}/di_ablation.csv")

    get_ablation_table(df)


if __name__ == "__main__":
    main()
