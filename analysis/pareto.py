import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from analysis.utils import (
    set_plt,
    MODELS_NAME_MAPPING,
    MODELS,
    MODELS_ARCH,
    MODELS_FIDS,
    MODELS_SIZES,
    MODELS_SINGLE_GENERATION,
    MODELS_NPASSES_TRAINING,
    MODELS_COMPUTE_TRAIN_FORWARD,
    MODELS_TIME_PER_SAMPLE,
    DMs,
)

MODELS_TRAINING_COST_TOTAL = {
    m: MODELS_NPASSES_TRAINING[m] * MODELS_COMPUTE_TRAIN_FORWARD[m] * 2
    for m in MODELS_NPASSES_TRAINING.keys()
}  # 2 because forward and backward pass

set_plt()

PATH_TO_DATA = "analysis/plots/pvalue_per_sample/tmp"
PATH_TO_PLOTS = "analysis/plots/pareto"


def get_data() -> pd.DataFrame:
    data = pd.concat(
        [
            pd.read_csv(os.path.join(PATH_TO_DATA, f))
            for f in tqdm(os.listdir(PATH_TO_DATA))
            if f != "pvalue_per_sample.csv" and f.endswith(".csv")
        ]
    )
    data = data.loc[data.Attack.isin(["LLM_MIA_CFG", "CDI"])]
    df = data.groupby(["Model", "n", "CLF"]).pvalue.mean().reset_index()

    dm_mask = df.Model.isin([MODELS_NAME_MAPPING[dm] for dm in DMs])
    df = pd.concat([df.loc[dm_mask & df.CLF], df.loc[~dm_mask & ~df.CLF]])

    priv_loss = df.loc[df.pvalue <= 0.01].groupby("Model").n.min().to_dict()
    mia_data = pd.read_csv(f"analysis/plots/mia_performance/mia_performance.csv")
    mia_data = (
        mia_data.groupby(["FTIDX", "Model"])["TPR@FPR=1\%"]
        .mean()
        .reset_index()
        .groupby("Model")["TPR@FPR=1\%"]
        .max()
        .to_dict()
    )

    out = []
    for model_raw in MODELS:
        model = MODELS_NAME_MAPPING[model_raw]
        out.append(
            {
                "Model": MODELS_ARCH[model_raw],
                "N samples": priv_loss.get(model, 10000),
                "FID": MODELS_FIDS[model_raw],
                "TPR@FPR=1\%": mia_data[model_raw],
                "Size": MODELS_SIZES[model_raw],
                "Class": "DM" if model_raw in DMs else "IAR",
                "Generation Cost": MODELS_SINGLE_GENERATION[model_raw] // 10**9,
                "Training Cost": MODELS_TRAINING_COST_TOTAL[model_raw] // 10**18,
                "Time per sample": MODELS_TIME_PER_SAMPLE[model_raw],
            }
        )
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(PATH_TO_PLOTS, "pareto.csv"), index=False)
    return df


def plot_data(
    df: pd.DataFrame,
    ax: plt.Axes,
    unique_archs: list,
    arch_color_map: dict,
    class_fill_map: dict,
    class_linestyle_map: dict,
    _100M_size: int,
    x: str,
    y: str,
):
    for arch in unique_archs:
        for cls in df["Class"].unique():
            subset = df[(df["Model"] == arch) & (df["Class"] == cls)]
            for _, row in subset.iterrows():
                area = row["Size"] / 10**8 * _100M_size
                markersize = np.sqrt(area)

                fill = class_fill_map[cls]
                facecolor = arch_color_map[arch] if fill == "full" else "none"
                edgewidth = 0.0 if fill == "full" else 2.0
                alpha = 0.5 if fill == "full" else 1.0
                edgecolor = arch_color_map[arch]

                ax.plot(
                    row[x],
                    row[y],
                    marker="o",
                    linestyle="None",
                    markersize=markersize,
                    fillstyle=fill,
                    markerfacecolor=facecolor,
                    markeredgecolor=edgecolor,
                    markeredgewidth=edgewidth,
                    alpha=alpha,
                )
    for arch in unique_archs:
        for cls in df["Class"].unique():
            subset = df[(df["Model"] == arch) & (df["Class"] == cls)]
            if len(subset) > 1:
                fill = class_fill_map[cls]
                subset_sorted = subset.sort_values(by=x)
                alpha = 0.5 if fill == "full" else 1.0
                ax.plot(
                    subset_sorted[x],
                    subset_sorted[y],
                    color=arch_color_map[arch],
                    linestyle=class_linestyle_map[cls],
                    linewidth=1.5,
                    alpha=alpha,
                )


def set_ax_di(ax: plt.Axes):
    ax.set_xscale("log")
    ax.set_xlim(2, 5 * 10**3)
    ax.set_ylim(1.30, 3.75)
    ax.set_xlabel("$\\mathit{P}$ (lower = less private)")
    ax.set_ylabel("FID (lower = better)")


def set_ax_tpr(ax: plt.Axes):
    ax.set_xscale("log")
    ax.set_xlim(1, 100)
    ax.set_ylim(1.30, 3.75)
    ax.set_xlabel("$\mathbf{TPR@FPR=1\%}$ (higher = less private)")
    ax.set_ylabel("FID (lower = better)")


def set_ax_gen(ax: plt.Axes):
    ax.set_xscale("log")
    ax.set_xlim(100, 300_000)
    ax.set_ylim(1.30, 3.75)
    ax.set_xlabel("Generaton cost (GFLOPs) (higher = costlier)")
    ax.set_ylabel("FID (lower = better)")


def set_ax_train(ax: plt.Axes):
    ax.set_xlim(10, 900)
    ax.set_ylim(1.30, 3.75)
    ax.set_xlabel("Training cost (1k PFLOPs) (higher = costlier)")
    ax.set_ylabel("FID (lower = better)")


def set_ax_time(ax: plt.Axes):
    ax.set_xscale("log")
    ax.set_xlim(0.05, 40)
    ax.set_ylim(1.30, 3.75)
    ax.set_xlabel("Latency (s/image) (lower = faster to generate)")
    ax.set_ylabel("FID (lower = better)")


def make_legend(
    arch_color_map: dict,
    class_fill_map: dict,
    size_map: dict,
    unique_archs: list,
    fig: plt.Figure,
):
    arch_legend_handles = []
    for idx, arch in enumerate(unique_archs):
        patch = mpatches.Patch(color=arch_color_map[arch], label=arch)
        if idx <= 2:
            patch.set_alpha(0.5)
        if not idx:
            arch_legend_handles.append(
                mlines.Line2D(
                    [], [], color="gray", linestyle="--", linewidth=1, label="IARs"
                )
            )
        arch_legend_handles.append(patch)
        if idx == 2:
            arch_legend_handles.append(
                mlines.Line2D(
                    [], [], color="gray", linestyle="--", linewidth=1, label="DMs"
                )
            )

    class_legend_handles = []
    for cls, fs in class_fill_map.items():
        line = mlines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="None",
            fillstyle=fs,
            markerfacecolor="black" if fs == "full" else "none",
            markeredgecolor="black",
            markersize=10,
            label=cls,
        )
        class_legend_handles.append(line)

    size_legend_handles = []
    for cat, area in size_map.items():
        ms = np.sqrt(area)
        line = mlines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="None",
            fillstyle="full",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=ms,
            label=cat,
        )
        size_legend_handles.append(line)

    all_legend_handles = (
        arch_legend_handles
        + [
            mlines.Line2D(
                [], [], color="gray", linestyle="--", linewidth=1, label="Architecture"
            )
        ]
        + class_legend_handles
        + [
            mlines.Line2D(
                [], [], color="gray", linestyle="--", linewidth=1, label="Size"
            )
        ]
        + size_legend_handles
        + [mlines.Line2D([], [], color="white", linestyle="--", linewidth=1, label="")]
        * 3
    )

    fig.legend(
        ncol=2,
        handles=all_legend_handles,
        title="Legend",
        loc="center left",
        borderpad=1.5,
        bbox_to_anchor=(1.0, 0.5),
        handlelength=3,
        handletextpad=1,
        handleheight=1.0,
        labelspacing=1.5,
    )


def plot_pareto(df: pd.DataFrame):
    unique_archs = df["Model"].unique()
    palette = sns.color_palette("tab10", len(unique_archs))
    arch_color_map = dict(zip(unique_archs, palette))

    _100M_size = 50

    class_fill_map = {"IAR": "full", "DM": "bottom"}
    class_linestyle_map = {"IAR": "-", "DM": ":"}

    df["SizeCat"] = pd.cut(
        df["Size"],
        bins=[0, 2 * 10**8, 6 * 10**8, 10**9, float("inf")],
        labels=["<200M", "200-600M", "600M-1B", ">1B"],
    )

    size_map = {
        "<200M": _100M_size,
        "200-600M": _100M_size * 4,
        "600M-1B": _100M_size * 8,
        ">1B": _100M_size * 15,
    }

    col_ax_mapping = {
        "N samples": set_ax_di,
        "TPR@FPR=1\%": set_ax_tpr,
        "Training Cost": set_ax_train,
        "Time per sample": set_ax_time,
        "Generation Cost": set_ax_gen,
    }

    for name, cols, (xsize, ysize) in zip(
        ["privacy", "speed", "teaser", "mia", "di"],
        [
            [["N samples", "TPR@FPR=1\%"]],
            [
                [
                    "Training Cost",
                    "Time per sample",
                ]
            ],
            [["TPR@FPR=1\%", "Time per sample"]],
            [["TPR@FPR=1\%"]],
            [["N samples"]],
        ],
        [
            (8, 5),
            (8, 5),
            (8, 5),
            (8, 5),
            (6, 5),
        ],
    ):
        cols_cnt = len(cols[0])
        rows_cnt = len(cols)
        fig, axs = plt.subplots(
            nrows=rows_cnt, ncols=cols_cnt, figsize=(xsize * cols_cnt, ysize * rows_cnt)
        )
        axs = axs.flatten() if rows_cnt * cols_cnt > 1 else [axs]

        for ax, x, y in zip(
            axs,
            np.array(cols).flatten(),
            ["FID"] * (cols_cnt * rows_cnt),
        ):
            plot_data(
                df,
                ax,
                unique_archs,
                arch_color_map,
                class_fill_map,
                class_linestyle_map,
                _100M_size,
                x=x,
                y=y,
            )
            col_ax_mapping[x](ax)

        make_legend(
            arch_color_map,
            class_fill_map,
            size_map,
            unique_archs,
            fig,
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(PATH_TO_PLOTS, f"pareto_{name}.png"), bbox_inches="tight"
        )
        plt.savefig(
            os.path.join(PATH_TO_PLOTS, f"pareto_{name}.pdf"), bbox_inches="tight"
        )


def main():
    os.makedirs(PATH_TO_PLOTS, exist_ok=True)
    df = get_data()
    plot_pareto(df)


if __name__ == "__main__":
    main()
