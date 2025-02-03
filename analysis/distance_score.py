import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

set_plt = lambda: plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.size": 15,  # Set font size to 11pt
        "axes.labelsize": 15,  # -> axis labels
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "lines.linewidth": 2,
        "text.usetex": False,
        "pgf.rcfonts": False,
    }
)
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
    df = pd.read_csv(f"{model}_memorized_train.csv")
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

# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(7, 5))

# Define colors for consistency
colors = {"var_30": "blue", "rar_xxl": "green", "mar_h": "red"}

# Loop through each model
for model in ["var_30", "rar_xxl", "mar_h"]:
    # Filter data
    tmp_df = df.loc[df.Model == model]

    # Compute correlation
    correlation = tmp_df[["cosine", "l2"]].corr().iloc[0, 1]

    # Scatter plot with low alpha
    ax.scatter(
        tmp_df.cosine,
        tmp_df.l2,
        label=f"{model} (ρ={correlation:.2f})",
        alpha=0.1,
        color=colors[model],
    )

    # Fit a linear regression line
    slope, intercept, _, _, _ = linregress(tmp_df.cosine, tmp_df.l2)
    x_vals = np.linspace(tmp_df.cosine.min(), tmp_df.cosine.max(), 100)
    y_vals = slope * x_vals + intercept
    ax.plot(x_vals, y_vals, color=colors[model], linewidth=2)

# Set labels
ax.set_xlabel(r"SSCD score", fontsize=12)
ax.set_ylabel(r"Distance $\mathit{d}$", fontsize=12)

# Customize legend to have full opacity
leg = ax.legend()
for lh in leg.legend_handles:
    lh.set_alpha(1)

# Show the plot
plt.savefig(
    "analysis/plots/memorization/memorization_distance_score.png",
    bbox_inches="tight",
    dpi=500,
)
