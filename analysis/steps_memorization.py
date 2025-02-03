import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from analysis.utils import set_plt
set_plt()

col = {
    "var_30": 4,
    "rar_xxl": 30,
}
indices = {
    "var_30": list(range(5)),
    "rar_xxl": [0, 1, 5, 14, 30],
}

dfs = []

for model in ["var_30", "rar_xxl"]:
    df = pd.read_csv(f"analysis/plots/memorization/{model}_memorized_train.csv")
    df["Model"] = model
    df = df[["Model", "sample_idx"] + [f"cosine_{idx}" for idx in indices[model]]]
    df.rename(
        columns={
            f"cosine_{idx}": f"cosine_{iidx}"
            for iidx, idx in zip([0, 1, 5, 14, 30], indices[model])
        },
        inplace=True,
    )

    dfs.append(df)

df = pd.concat(dfs)
dfs = []
for col in df.columns[2:]:
    tmp_df = df.loc[df[col] > 0.75].groupby("Model").sample_idx.count().reset_index()
    tmp_df["Tokens"] = int(col.split("_")[-1])
    dfs.append(tmp_df)

df = pd.concat(dfs)
df.replace(
    {"mar_h": "MAR-H", "rar_xxl": "RAR-XXL", "var_30": "VAR-$\\mathit{d}$30"},
    inplace=True,
)

_, ax = plt.subplots(1, 1, figsize=(7, 5))
df.pivot(index="Tokens", columns="Model", values="sample_idx").plot.line(ax=ax)
ax.set_xlabel("Prefix length $i$")
ax.set_ylabel("Number of memorized samples")
plt.savefig("analysis/plots/memorization/prefix_length.pdf", bbox_inches="tight")
