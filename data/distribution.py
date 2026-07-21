import argparse
import os
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "legend.title_fontsize": 18,
})

def main():
    #distribution of data in regards to EC, Organisms
    #distribution of log KCat/KM values in regards to EC 
    data_path = "data/data_KCATKM.csv"
    df = pd.read_csv(data_path)



    df["EC_class"] = df["EC"].astype(str).str.split(".", n=1).str[0]
    #print(df["EC_class"])
    counts = df["EC_class"].value_counts().sort_index()
    labels = counts.index.tolist()
    sizes = counts.values

    fig, ax = plt.subplots(figsize = (8,8))
    palette = sns.color_palette("Set2", n_colors=len(labels))
    wedges, texts = ax.pie(
        sizes,
        labels=None,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(edgecolor="w"),
        colors=palette
    )
    total = sizes.sum()
    legend_labels = [f"{lab}: {cnt} ({cnt/total*100:.1f}%)" for lab, cnt in zip(labels, sizes)]

    ax.legend(
        wedges,
        legend_labels,
        title="EC class (count, %)",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        fontsize=16,
        title_fontsize=16,
    )

    ax.set_title("EC class distribution")
    ax.axis("equal")
    plt.tight_layout()
    out = Path(__file__).parent / "ec_pie.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

 

    org_counts = df["ORGANISM"].value_counts().head(15)
    labels = org_counts.index.tolist()
    plt.figure(figsize=(10, 6))
    ax = org_counts.plot(kind="barh", color="skyblue", edgecolor="black", linewidth=1)
    ax.set_yticklabels(labels, fontstyle="italic")
    plt.xlabel(r"$k_{\mathrm{cat}}/K_{\mathrm{m}}$ Count")
    plt.ylabel("")
    plt.title(r"$k_{\mathrm{cat}}/K_{\mathrm{m}}$ Distribution by Organisms")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    out = Path(__file__).parent / "organism.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()



    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=df,
        x="EC_class",
        y="Log10_value",
        hue="EC_class",
        palette="Set2",
        legend=False
    )
    plt.title("$\log_{10}(k_{\mathrm{cat}}/K_{\mathrm{m}})$ by EC class")
    plt.xlabel("EC class")
    plt.ylabel(" $\log_{10}(k_{\mathrm{cat}}/K_{\mathrm{m}})$ values")
    plt.xticks(rotation=0)
    plt.tight_layout()
    out = Path(__file__).parent / "violin_EC.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
if __name__ == '__main__':
    main()