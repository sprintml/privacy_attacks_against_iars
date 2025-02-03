import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.stats import linregress

from analysis.utils import set_plt

set_plt()
col = {
    "var_30": 4,
    "rar_xxl": 30,
    "mar_h": 5,
}
indices = {
    "var_30": list(range(5)),
    "rar_xxl": [0, 1, 5, 14, 30],
    "mar_h": [0, 1, 5],
}

dfs = []

for model in ["var_30", "rar_xxl", "mar_h"]:
    df = pd.read_csv(f"analysis/plots/memorization/{model}_memorized_train.csv")
    df["Model"] = model
    df = df[["Model"] + ["sample_idx", f"cosine_{col[model]}", f"l2_{col[model]}"]]
    df.rename(
        columns={
            f"cosine_{col[model]}": "cosine",
            f"l2_{col[model]}": "l2",
            **{f"token_eq_{idx}": f"d" for i, idx in enumerate(indices[model])},
        },
        inplace=True,
    )

    dfs.append(df)

df = pd.concat(dfs)
distances = pd.read_csv("analysis/plots/memorization/mem_dist.csv")
dfs = []
for model, mapping in zip(["var_30", "rar_xxl", "mar_h"], ["VAR", "RAR", "MAR"]):
    tmp_df = df.loc[df.Model == model]
    tmp_distances = distances[mapping]
    tmp_df.l2 = tmp_df.apply(lambda x: tmp_distances[int(x.sample_idx)], axis=1)
    if model == "var_30" or model == "rar_xxl":
        tmp_df.l2 = tmp_df.l2.apply(lambda x: 100 * (1 - x / 256)).copy()
    dfs.append(tmp_df)
df = pd.concat(dfs)

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
colors = {"var_30": "blue", "rar_xxl": "green", "mar_h": "red"}
for model in ["var_30", "rar_xxl", "mar_h"]:
    tmp_df = df.loc[df.Model == model]

    correlation = tmp_df[["cosine", "l2"]].corr().iloc[0, 1]
    ax.scatter(
        tmp_df.cosine,
        tmp_df.l2,
        label=f"{model} (ρ={correlation:.2f})",
        alpha=0.1,
        color=colors[model],
    )

    slope, intercept, _, _, _ = linregress(tmp_df.cosine, tmp_df.l2)
    x_vals = np.linspace(tmp_df.cosine.min(), tmp_df.cosine.max(), 100)
    y_vals = slope * x_vals + intercept
    ax.plot(x_vals, y_vals, color=colors[model], linewidth=2)

ax.set_xlabel(r"SSCD score", fontsize=12)
ax.set_ylabel(r"Distance $\mathit{d}$", fontsize=12)

leg = ax.legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)

plt.savefig(
    "analysis/plots/memorization/memorization_distance_score.png",
    bbox_inches="tight",
    dpi=500,
)
