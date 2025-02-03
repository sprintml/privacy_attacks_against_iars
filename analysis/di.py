import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from itertools import product
import multiprocessing as mp
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from analysis.evaluate import get_p_value, get_datasets_clf
from analysis.utils import (
    set_plt,
    ATTACKS_NAME_MAPPING,
    MODELS_NAME_MAPPING,
    MODELS_COLORS,
    MODELS_ORDER,
    MODELS_STYLES,
    RESAMPLING_CNT,
    ATTACKS,
    MODELS,
    RUN_ID,
    OURS,
    DMs,
    IARs,
)
import pandas as pd

from typing import Tuple

set_plt()

N_PROC = 80

MODELS = IARs
ATTACKS = ["llm_mia_loss", "llm_mia_cfg"]


DO_CLF = [False]
PVALUES = [0.01, 0.05]
NSAMPLES_TO_SHOW = [10, 100, 500, 1000, 5000, 10000]
REMOVE_SURP = True
REMOVE_CAMIA = True

PATH_TO_FEATURES = "out/features"
PATH_TO_PLOTS = "analysis/plots/pvalue_per_sample"

SURP_IDX = np.arange(8, 48)
CAMIA_IDX_MAR = np.arange(7, 10)


def remove_features(features, SURP_IDX):
    filtered_features = features[
        :, :, np.setdiff1d(np.arange(features.shape[2]), SURP_IDX)
    ]
    return filtered_features


def plot_pvalues_cmp(
    title: str,
    df: pd.DataFrame,
    ax: plt.Axes,
    hue: str,
    xlabel: str = "Number of samples",
):
    sns.lineplot(
        data=df,
        x="n",
        y="pvalue",
        hue=hue,
        ax=ax,
        palette=(
            [MODELS_COLORS[model] for model in MODELS_ORDER] if hue == "Model" else None
        ),
    )
    ax.plot(df.n, 0.05 * np.ones_like(df.n), "--", color="black", label="p-value: 0.05")
    ax.plot(df.n, 0.01 * np.ones_like(df.n), "--", color="green", label="p-value: 0.01")
    ax.set(
        xscale="log",
        yscale="log",
        ylim=[10 ** (-3), 1],
        title=title,
        ylabel="p-value",
        xlabel=xlabel,
    )
    if title is not None:
        ax.legend(loc="lower left")
    else:
        ax.get_legend().remove()


def plot_pvalues_ours(
    title: str,
    df: pd.DataFrame,
    ax: plt.Axes,
    hue: str,
    xlabel: str = "Number of samples",
):

    for idx, model in enumerate(MODELS_ORDER):
        model_data = df[df[hue] == model]
        sns.lineplot(
            data=model_data,
            x="n",
            y="pvalue",
            ax=ax,
            color=MODELS_COLORS[model],
            linestyle=MODELS_STYLES[model],
            legend=idx == len(MODELS_ORDER) - 1,
            label=model,
        )

    ax.plot(df.n, 0.05 * np.ones_like(df.n), "--", color="black", label="p-value: 0.05")
    ax.plot(df.n, 0.01 * np.ones_like(df.n), "--", color="green", label="p-value: 0.01")
    ax.set(
        xscale="log",
        yscale="log",
        ylim=[10 ** (-3), 1],
        title=title,
        ylabel="p-value",
        xlabel=xlabel,
    )
    if title is not None:
        ax.legend(loc="lower left")
    else:
        ax.get_legend().remove()


def plot_min_n_pvalue(df: pd.DataFrame, ax: plt.Axes, pvalue: int):
    df.loc[df.pvalue <= pvalue].groupby(["Attack", "Model"]).n.min().unstack(
        level=0
    ).plot(kind="bar", ax=ax, width=0.9, edgecolor="black", linewidth=1, legend=False)

    for i in range(len(ax.containers)):
        ax.bar_label(
            ax.containers[i], fmt="%.0f", label_type="edge", rotation=90, padding=5
        )

    ax.set(
        xlabel="Model",
        ylabel="Number of samples",
        title=f"Minimum number of samples for p-value <= {pvalue}",
    )


def get_ca_table(df: pd.DataFrame) -> pd.DataFrame:
    pvalue_at_k_df = (
        df.groupby(["Attack", "Model", "n"])
        .pvalue.apply(lambda x: f"{x.mean():.1e}$\pm${x.std():.1e}")
        .reset_index()
    )
    pvalue_at_k_df = pvalue_at_k_df.loc[pvalue_at_k_df.n.isin(NSAMPLES_TO_SHOW)].pivot(
        index="n", columns="Model", values="pvalue"
    )
    return pvalue_at_k_df.loc[NSAMPLES_TO_SHOW][MODELS_ORDER]


def preprocess_features(
    members: np.ndarray, nonmembers: np.ndarray, do_clf: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    B = members.shape[0]

    members[np.isnan(members)] = 0
    nonmembers[np.isnan(nonmembers)] = 0

    if do_clf:
        return get_classifier_scores(members, nonmembers)

    scaler = MinMaxScaler()
    data = np.concat([members.reshape(B, -1), nonmembers.reshape(B, -1)], axis=0)
    data = scaler.fit_transform(data)

    members = data[: members.shape[0]]
    nonmembers = data[members.shape[0] :]
    return members.sum(axis=1), nonmembers.sum(axis=1)


def get_classifier_scores(
    members: np.ndarray,
    nonmembers: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    members_indices = np.arange(len(members))
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    members_scores, nonmembers_scores = [], []

    for train_idx, test_idx in kf.split(members_indices):
        # Get train and test splits for members and nonmembers using the same indices
        members_train = torch.from_numpy(members[train_idx, 0])
        members_test = torch.from_numpy(members[test_idx, 0])
        nonmembers_train = torch.from_numpy(nonmembers[train_idx, 0])
        nonmembers_test = torch.from_numpy(nonmembers[test_idx, 0])

        # Create training and testing datasets using get_datasets_clf
        train_dataset = get_datasets_clf(members_train, nonmembers_train)
        test_dataset = get_datasets_clf(members_test, nonmembers_test)

        # Standardize the data
        ss = StandardScaler()
        train_data = ss.fit_transform(train_dataset.data)
        test_data = ss.transform(test_dataset.data)

        # Train the classifier
        clf = LogisticRegression(random_state=0, max_iter=1000, n_jobs=None)
        clf.fit(train_data, train_dataset.label)

        # Compute scores only for the test dataset
        scores = clf.predict_proba(test_data)[:, 1]

        # Separate the scores for members and nonmembers
        members_scores.append(scores[: len(test_dataset.data) // 2])
        nonmembers_scores.append(scores[len(test_dataset.data) // 2 :])

    members_scores = np.concat(members_scores)
    nonmembers_scores = np.concat(nonmembers_scores)

    assert (
        len(members_scores)
        == len(nonmembers_scores)
        == (len(test_dataset.data) + len(train_dataset.data)) // 2
    )
    assert members_scores.shape == nonmembers_scores.shape
    assert len(members_scores.shape) == 1

    # negate so members are lower
    return -members_scores, -nonmembers_scores


def get_scores():
    all_members, all_nonmembers = [], []
    for attack, model, _ in product(ATTACKS, MODELS, DO_CLF):
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
    do_clf: bool,
    n: int,
    indices_total: np.ndarray,
    members: np.ndarray,
    nonmembers: np.ndarray,
    members_lower: bool,
):
    out = []
    for r in tqdm(
        range(RESAMPLING_CNT),
        desc=f"{model} {attack} {do_clf=} {n=}",
        total=RESAMPLING_CNT,
    ):
        try:
            indices = np.random.choice(indices_total, n, replace=False)
            members_eval, nonmembers_eval = (
                members[indices].copy(),
                nonmembers[indices].copy(),
            )
            members_scores, nonmembers_scores = preprocess_features(
                members_eval, nonmembers_eval, do_clf=do_clf
            )
            pvalue, is_correct_order = get_p_value(
                members_scores, nonmembers_scores, members_lower=members_lower
            )
            out.append(
                [
                    ATTACKS_NAME_MAPPING[attack],
                    MODELS_NAME_MAPPING[model],
                    do_clf,
                    n,
                    pvalue,
                    is_correct_order,
                    r,
                ]
            )
        except Exception as e:
            print("Exception", model, attack, do_clf, n, r, e)
            exit()
    pd.DataFrame(
        out,
        columns=["Attack", "Model", "CLF", "n", "pvalue", "is_correct_order", "r"],
    ).to_csv(
        f"{PATH_TO_PLOTS}/tmp/pvalue_per_sample_{model}_{attack}_{'clf' if do_clf else 'sum'}_{n}.csv",
        index=False,
    )
    print(f"Done {model} {attack}, {do_clf=}, {n=}")


def get_data() -> list:
    # [n for n in range(2, 11, 1)]
    # [n for n in range(20, 110, 10)]
    nsamples = np.array(
        [n for n in range(2, 11, 1)]
        + [n for n in range(20, 110, 10)]
        + [n for n in range(200, 1100, 100)]
        + [n for n in range(2000, 11_000, 1000)]
    )
    # nsamples = np.array([n for n in range(2000, 11_000, 1000)])
    indices_total = np.arange(10_000)

    members, nonmembers = get_scores()
    print("Scores loaded")

    with mp.Pool(N_PROC) as pool:
        for idx, (attack, model, do_clf) in enumerate(product(ATTACKS, MODELS, DO_CLF)):
            m, nm, ml = members[idx][0], nonmembers[idx][0], members[idx][1]
            if m is None:
                continue
            for n in nsamples:
                pool.apply_async(
                    get_row,
                    args=(model, attack, do_clf, n, indices_total, m, nm, ml),
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


def get_pvalue_per_sample_cmp(df: pd.DataFrame):
    _, axs = plt.subplots(
        1,
        len(MODELS),
        figsize=(10 * len(MODELS), 5 * 1),
    )
    axs = axs.flatten()

    for idx, model in tqdm(enumerate(MODELS)):
        model = MODELS_NAME_MAPPING[model]
        tmp_df = df[df["Model"] == model]
        plot_pvalues_cmp(model, tmp_df, axs[idx], hue="Attack")

    plt.savefig(
        f"{PATH_TO_PLOTS}/pvalue_per_sample.pdf", format="pdf", bbox_inches="tight"
    )


def get_min_n_samples_cmp(df: pd.DataFrame):
    fig, axs = plt.subplots(
        1,
        len(PVALUES),
        figsize=(20 * len(PVALUES), 5 * 1),
    )

    df = df.groupby(["Attack", "Model", "n"]).pvalue.mean().reset_index()
    for idx, (ax, pvalue) in tqdm(enumerate(zip(axs, PVALUES))):
        plot_min_n_pvalue(df, ax, pvalue)

    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(ATTACKS),
        title="Attack",
    )
    plt.savefig(f"{PATH_TO_PLOTS}/min_n_pvalue.pdf", format="pdf", bbox_inches="tight")


def main():
    os.makedirs(os.path.join(PATH_TO_PLOTS, "tmp"), exist_ok=True)
    np.random.seed(42)
    try:
        df = pd.read_csv(f"{PATH_TO_PLOTS}/pvaluae_per_sample.csv")
    except FileNotFoundError:
        data = get_data()
        exit()
        df = pd.DataFrame(
            data,
            columns=["Attack", "Model", "CLF", "n", "pvalue", "is_correct_order", "r"],
        )
        df.to_csv(f"{PATH_TO_PLOTS}/pvalue_per_sample.csv", index=False)

    # get_ca_table(df.loc[df.Attack == OURS]).to_latex(
    #     f"{PATH_TO_PLOTS}/ca_table.tex", escape=False
    # )

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10 * 1, 3 * 1),
    )

    plot_pvalues_ours(
        None,
        df.loc[(df.Attack == OURS) & (df.CLF == False)],
        ax,
        hue="Model",
        xlabel="Number of samples in $\mathbf{P}$",
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        ncols=2,
        title="Model",
        bbox_to_anchor=(1.25, 0.5),
    )
    plt.savefig(
        f"{PATH_TO_PLOTS}/pvalue_per_sample_{OURS}.pdf",
        format="pdf",
        bbox_inches="tight",
    )

    get_pvalue_per_sample_cmp(df.loc[df.CLF == False])
    get_min_n_samples_cmp(df)


if __name__ == "__main__":
    main()
