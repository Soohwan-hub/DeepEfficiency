from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "legend.title_fontsize": 18,
})


def _annotate_bars(ax, decimals=3, zero_tol=1e-12):
    for container in ax.containers:
        labels = []
        for bar in container:
            height = bar.get_height()
            if np.isnan(height) or np.isclose(height, 0.0, atol=zero_tol):
                labels.append("")
            else:
                labels.append(f"{height:.{decimals}f}")
        ax.bar_label(container, labels=labels, padding=2, fontsize=18)


def _load_metrics():
    xgboost_best = Path(
        "results/xgboostcpu/run_2026-07-06_23-48-42/summary_report_2026-07-07_09-45-53.csv"
    )
    unikp = Path("results/unikp(3.10)/Kinetic_parameters_metrics.csv")
    catapro = Path("results/catapro/CataPro_metrics.csv")

    xgboost_df = pd.read_csv(xgboost_best)
    unikp_df = pd.read_csv(unikp)
    catapro_df = pd.read_csv(catapro)

    metrics = pd.DataFrame(
        {
            "model": ["XGBoost", "UniKP", "CataPro"],
            "$R^2$": [
                float(xgboost_df.loc[0, "test_r2"]),
                float(unikp_df.loc[0, "r2"]),
                float(catapro_df.loc[0, "r2"]),
            ],
            "pearson": [
                float(xgboost_df.loc[0, "test_pearson"]),
                float(unikp_df.loc[0, "pearson"]),
                float(catapro_df.loc[0, "pearson"]),
            ],
        }
    )
    return metrics


def comparison_bargraph(output_path="results/model_comparison.png"):
    metrics = _load_metrics()
    metrics_long = metrics.melt(id_vars="model", var_name="metric", value_name="value")
    metrics_long["metric"] = metrics_long["metric"].replace({
        "r2": r"$R^2$",
        "pearson": "Pearson",
    })

    sns.set_style("white")
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=metrics_long,
        x="model",
        y="value",
        hue="metric",
        edgecolor="black",
        linewidth=1
    )
    ax.grid(False)
    _annotate_bars(ax)

    plt.ylabel("Scores")
    plt.xlabel("")
    plt.title(r"$R^2$ and Pearson by Models")
    plt.ylim(0, 1)
    plt.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {output}")


if __name__ == "__main__":
    comparison_bargraph()