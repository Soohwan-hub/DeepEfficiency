import argparse
import os
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

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

    ax.legend(wedges, legend_labels, title="EC class (count, %)",
            bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=9)

    ax.set_title("EC class distribution")
    ax.axis("equal")
    plt.tight_layout()
    out = Path(__file__).parent / "ec_pie.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()



    org_counts = df["ORGANISM"].value_counts().head(15)
    labels = org_counts.index.tolist()
    plt.figure(figsize=(10, 6))
    org_counts.plot(kind="barh", color="skyblue", edgecolor="black")
    plt.xlabel("Count")
    plt.ylabel("Organism")
    plt.title("Top 15 Organisms")
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
        y="Log10 KCat/KM value",
        hue="EC_class",
        palette="Set2",
        legend=False
    )
    plt.title("Violin plot of Log10_value by EC class")
    plt.xlabel("EC class")
    plt.ylabel("Log10_value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    out = Path(__file__).parent / "violin_EC.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
if __name__ == '__main__':
    main()