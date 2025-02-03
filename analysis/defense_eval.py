import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


from analysis.utils import (
    set_plt,
    IARs,
    MODELS_ORDER,
    MODELS_STYLES,
    MODELS_NAME_MAPPING,
    MODELS_COLORS,
)

from typing import Tuple, List

set_plt()

RUN_ID = "10k"
MODELS = IARs
N_PROC = 20

mia_df = pd.read_csv("analysis/plots/def_mia_performance/def_mia_performance_agg.csv")
fid_df = pd.read_csv("analysis/plots/defense/fids.csv")
mem_df = pd.read_csv("analysis/plots/defense/memorization.csv")
di_df = pd.read_csv("analysis/plots/def_di_performance/pvalue_per_sample_agg.csv")
PATH_TO_PLOTS = "analysis/plots/defense_eval"


def plot(
    df: pd.DataFrame,
    ax: plt.Axes,
    ycol: str,
    ylabel: str,
    ylog: bool,
    title: str,
    ylim: Tuple[float, float],
    legend: bool = False,
    xlabel: str = "$\sigma$",
):
    print(ycol, ylabel, ylog, title, ylim, xlabel, legend)
    df.Model = df.Model.map(MODELS_NAME_MAPPING)
    for idx, model in enumerate(MODELS_ORDER):
        model_data = df[df.Model == model]
        model_data = model_data.copy()
        model_data["STD"] = model_data["STD"].astype(str)
        model_data = model_data.sort_values(by="STD", key=lambda x: x.astype(float))
        sns.lineplot(
            data=model_data,
            x="STD",
            y=ycol,
            ax=ax,
            color=MODELS_COLORS[model],
            linestyle=MODELS_STYLES[model],
            legend=legend,
            label=model,
        )

    ax.set(
        xscale="linear",
        yscale="log" if ylog else "linear",
        ylim=ylim,
        title=title,
        ylabel=ylabel,
        xlabel=xlabel,
    )
    xticks = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
    ax.set_xticks(np.linspace(0, len(xticks) - 1, len(xticks)))
    ax.set_xticklabels([str(x) for x in xticks])
    if legend:
        ax.legend().remove()


def main():
    os.makedirs(os.path.join(PATH_TO_PLOTS, "tmp"), exist_ok=True)
    np.random.seed(42)
    fig, axs = plt.subplots(1, 4, figsize=(24, 4))

    ycols = ["TPR@FPR=1\%", "n", "n", "FID"]

    ylabels = [
        "TPR@FPR=1\%($\downarrow$)",
        "$\\mathit{P}(\\uparrow$)",
        "Count($\downarrow$)",
        "FID($\downarrow$)",
    ]
    ylogs = [True, True, True, False]
    titles = [
        "Membership Inference Attack",
        "Dataset Inference",
        "Extraction Attack",
        "Generation Quality",
    ]
    ylims = [
        (0.9, 100),
        (5, 3000),
        (1, 1000),
        (4, 10),
    ]
    legends = [False, False, False, True]

    for idx, (df, ax, ycol, ylabel, ylog, title, ylim, legend) in enumerate(
        zip(
            [mia_df, di_df, mem_df, fid_df],
            axs,
            ycols,
            ylabels,
            ylogs,
            titles,
            ylims,
            legends,
        )
    ):
        plot(df, ax, ycol, ylabel, ylog, title, ylim, legend)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles[: len(IARs)],
        labels[: len(IARs)],
        loc="center right",
        ncols=1,
        title="Model",
        bbox_to_anchor=(0.975, 0.5),
    )
    plt.savefig(
        f"{PATH_TO_PLOTS}/defense_eval.pdf",
        format="pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
