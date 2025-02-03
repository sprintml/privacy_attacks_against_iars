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
    MODELS_ORDER,
    MODELS,
)

from typing import List

MODELS_ORDER = ["mar_b", "mar_l", "mar_h"]
MODELS = ["mar_b", "mar_l", "mar_h"]

CONFIGS = [
    {
        "RUN_IDs": [
            "10k_ratio0.70",
            "10k_ratio0.75",
            "10k_ratio0.80",
            "10k_ratio0.85",
            "10k_ratio0.90",
            "10k_ratio0.95",
            "10k_ratio0.99",
        ],
        "RUN_ID_NAME_MAPPING": {
            run_id: run_id.split("_")[-1]
            for run_id in [
                "10k_ratio0.70",
                "10k_ratio0.75",
                "10k_ratio0.80",
                "10k_ratio0.85",
                "10k_ratio0.90",
                "10k_ratio0.95",
                "10k_ratio0.99",
            ]
        },
        "REFERENCE_ID": "10k_ratio0.90",
    },
    {
        "RUN_IDs": ["10k_ratio0.95", "10k_4_ratio0.95"],
        "RUN_ID_NAME_MAPPING": {"10k_ratio0.95": "1", "10k_4_ratio0.95": "4"},
        "REFERENCE_ID": "10k_ratio0.95",
    },
    {
        "RUN_IDs": [
            "10k_4_ratio0.95",
            "timestep_10k_t100",
            "timestep_10k_t300",
            "timestep_10k_t500",
            "timestep_10k_t700",
            "timestep_10k_t900",
        ],
        "RUN_ID_NAME_MAPPING": {
            "10k_4_ratio0.95": "random",
            "timestep_10k_t100": "100",
            "timestep_10k_t300": "300",
            "timestep_10k_t500": "500",
            "timestep_10k_t700": "700",
            "timestep_10k_t900": "900",
        },
        "REFERENCE_ID": "10k_4_ratio0.95",
    },
    {
        "RUN_IDs": [
            "timestep_10k_t700",
            "timestep_10k_mul8_t700",
            "timestep_10k_mul16_t700",
            "timestep_10k_mul32_t700",
            "timestep_10k_mul64_t700",
        ],
        "RUN_ID_NAME_MAPPING": {
            "timestep_10k_t700": "4",
            "timestep_10k_mul8_t700": "8",
            "timestep_10k_mul16_t700": "16",
            "timestep_10k_mul32_t700": "32",
            "timestep_10k_mul64_t700": "64",
        },
        "REFERENCE_ID": "timestep_10k_t700",
    },
    {
        "RUN_IDs": [
            "10k_ratio0.90",
            "10k_ratio0.95",
            "10k_4_ratio0.95",
            "timestep_10k_t700",
            "timestep_10k_mul64_t700",
        ],
        "RUN_ID_NAME_MAPPING": {
            "10k_ratio0.90": "Baseline",
            "10k_ratio0.95": "+ Adjusted Binary Mask",
            "10k_4_ratio0.95": "+ Multiple Masks",
            "timestep_10k_t700": "+ Fixed Timestep",
            "timestep_10k_mul64_t700": "+ Increased Multiplication",
        },
        "REFERENCE_ID": None,
    },
]

set_plt()

SIZE = 5_000
PATH_TO_FEATURES = "out/features"
PATH_TO_PLOTS = "analysis/plots/mar_mia_ablation"


def get_data(RUN_IDs: List[str]) -> List:
    out = []
    for model, run_id in tqdm(product(MODELS, RUN_IDs)):
        np.random.seed(42)
        try:
            members = np.load(
                f"{PATH_TO_FEATURES}/{model}_llm_mia_loss_{run_id}_imagenet_train.npz",
                allow_pickle=True,
            )["data"]
            nonmembers = np.load(
                f"{PATH_TO_FEATURES}/{model}_llm_mia_loss_{run_id}_imagenet_val.npz",
                allow_pickle=True,
            )["data"]
        except FileNotFoundError:
            print(
                f"Missing {model}, {PATH_TO_FEATURES}/{model}_llm_mia_loss_{run_id}_imagenet_val.npz"
            )
            continue
        members_lower = True
        for ft_idx, _ in product(range(members.shape[2]), range(250)):
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
                    model,
                    run_id,
                    f"{tpr*100:.2f}",
                    f"{auc*100:.2f}",
                    f"{accuracy*100:.2f}",
                ]
            )

    return out


def get_ablation_table(
    data: pd.DataFrame, RUN_IDs: List[str], RUN_ID_NAME_MAPPING: dict, REFERENCE_ID: str
):
    data = (
        data.groupby(["Model", "FTIDX", "Run ID"])["TPR@FPR=1\%"].mean().reset_index()
    )
    data = data.groupby(["Model", "Run ID"])["TPR@FPR=1\%"].max().reset_index()
    data = data.pivot(index="Run ID", columns="Model", values="TPR@FPR=1\%")

    # Reorder columns based on MODELS_ORDER
    data = data[[model for model in MODELS_ORDER if model in data.columns]]

    formatted_data = pd.DataFrame(index=data.index, columns=data.columns)

    # First row remains as is
    formatted_data.loc[data.index == RUN_IDs[0]] = data.loc[
        data.index == RUN_IDs[0]
    ].applymap(lambda x: f"{x:.2f}")

    if REFERENCE_ID:
        # Compute improvement and format cells
        for run in RUN_IDs:
            value = data.loc[data.index == run]
            improvement = value - data.loc[REFERENCE_ID]
            formatted_data.loc[data.index == run] = value.applymap(
                lambda x: f"{x:.2f}"
            ) + improvement.applymap(lambda x: f" ({'+' if x > 0 else ''}{x:.2f})")
    else:
        for run_prev, run in zip(RUN_IDs[:-1], RUN_IDs[1:]):
            value = data.loc[data.index == run]
            improvement = value - data.loc[run_prev]
            formatted_data.loc[data.index == run] = value.applymap(
                lambda x: f"{x:.2f}"
            ) + improvement.applymap(lambda x: f" ({'+' if x > 0 else ''}{x:.2f})")

    # Rename indices
    formatted_data.index = formatted_data.index.map(RUN_ID_NAME_MAPPING)

    print(formatted_data)
    # Save to LaTeX
    formatted_data.to_latex(
        f"{PATH_TO_PLOTS}/mar_ablation_{REFERENCE_ID}.tex", escape=False
    )

    return formatted_data


def main(RUN_IDs: List[str], RUN_ID_NAME_MAPPING: dict, REFERENCE_ID: str):
    os.makedirs(PATH_TO_PLOTS, exist_ok=True)
    np.random.seed(42)

    data = get_data(RUN_IDs)
    df = pd.DataFrame(
        data, columns=["FTIDX", "Model", "Run ID", "TPR@FPR=1\%", "AUC", "Accuracy"]
    )
    df.to_csv(f"{PATH_TO_PLOTS}/mia_performance_{REFERENCE_ID}.csv", index=False)
    df = pd.read_csv(f"{PATH_TO_PLOTS}/mia_performance_{REFERENCE_ID}.csv")

    get_ablation_table(df, RUN_IDs, RUN_ID_NAME_MAPPING, REFERENCE_ID)


if __name__ == "__main__":
    for c in tqdm([CONFIGS[-1]]):
        RUN_IDs = c["RUN_IDs"]
        RUN_ID_NAME_MAPPING = c["RUN_ID_NAME_MAPPING"]
        REFERENCE_ID = c["REFERENCE_ID"]
        main(RUN_IDs, RUN_ID_NAME_MAPPING, REFERENCE_ID)
