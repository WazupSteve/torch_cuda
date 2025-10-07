"""
Descriptive Analysis Module

Performs descriptive statistics, correlation analysis, and ANOVA.
"""

from typing import Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path


class DescriptiveAnalysis:
    def __init__(self, data_path: str = "data/processed/forum_data.csv"):
        self.df = pd.read_csv(data_path)
        self.output_dir = Path("data/processed/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)

    def summary_statistics(self) -> pd.DataFrame:
        comparison = (
            self.df.groupby("is_cuda_related")
            .agg({
                "time_to_resolution_hours": ["mean", "median", "std", "count"],
                "time_to_first_response_hours": ["mean", "median"],
                "is_resolved": "mean",
                "views": "mean",
                "reply_count": "mean",
                "has_code_block": "mean",
                "question_length": "mean",
            })
            .round(2)
        )
        return comparison

    def correlation_analysis(self) -> pd.DataFrame:
        numerical_cols = [
            "is_cuda_related", "views", "reply_count", "like_count", "has_code_block",
            "code_block_count", "question_length", "has_error_trace",
            "time_to_resolution_hours", "time_to_first_response_hours",
        ]
        numerical_cols = [col for col in numerical_cols if col in self.df.columns]

        corr_matrix = self.df[numerical_cols].corr().round(3)

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, square=True)
        plt.title("Correlation Matrix", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.output_dir / "correlation_matrix.png", dpi=300)
        plt.close()

        return corr_matrix

    def anova_category_effect(self) -> Dict:
        resolved = self.df[self.df["time_to_resolution_hours"].notna()].copy()
        categories = resolved.groupby("category_id")["time_to_resolution_hours"].apply(list)
        categories = {k: v for k, v in categories.items() if len(v) >= 5}
        f_stat, p_value = stats.f_oneway(*categories.values())
        return {"f_statistic": f_stat, "p_value": p_value}

    def plot_distributions(self):
        resolved = self.df[self.df["time_to_resolution_hours"].notna()]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Resolution time histogram
        axes[0, 0].hist(resolved["time_to_resolution_hours"], bins=50, edgecolor="black", alpha=0.7)
        axes[0, 0].set_xlabel("Time to Resolution (hours)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Distribution of Resolution Times")

        # CUDA vs Non-CUDA boxplot
        cuda_resolved = resolved[resolved["is_cuda_related"]]
        non_cuda_resolved = resolved[~resolved["is_cuda_related"]]
        axes[0, 1].boxplot([non_cuda_resolved["time_to_resolution_hours"], cuda_resolved["time_to_resolution_hours"]],
                          labels=["Non-CUDA", "CUDA"])
        axes[0, 1].set_ylabel("Time to Resolution (hours)")
        axes[0, 1].set_title("Resolution Time: CUDA vs Non-CUDA")

        # Views distribution
        axes[1, 0].hist(self.df["views"], bins=50, edgecolor="black", alpha=0.7, log=True)
        axes[1, 0].set_xlabel("Views")
        axes[1, 0].set_ylabel("Frequency (log scale)")
        axes[1, 0].set_title("Distribution of Views")

        # Resolution rate by CUDA status
        resolution_rates = self.df.groupby("is_cuda_related")["is_resolved"].mean()
        axes[1, 1].bar(["Non-CUDA", "CUDA"], resolution_rates.values, color=["skyblue", "coral"], edgecolor="black")
        axes[1, 1].set_ylabel("Resolution Rate")
        axes[1, 1].set_title("Resolution Rate: CUDA vs Non-CUDA")
        axes[1, 1].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(self.output_dir / "distributions.png", dpi=300)
        plt.close()

    def plot_scatter_views_vs_resolution(self):
        resolved = self.df[self.df["time_to_resolution_hours"].notna()]
        plt.figure(figsize=(10, 6))
        colors = resolved["is_cuda_related"].map({True: "coral", False: "skyblue"})
        plt.scatter(resolved["views"], resolved["time_to_resolution_hours"], c=colors, alpha=0.5, edgecolors="black")
        plt.xlabel("Views")
        plt.ylabel("Time to Resolution (hours)")
        plt.title("Views vs Resolution Time")
        plt.xscale("log")

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="skyblue", edgecolor="black", label="Non-CUDA"),
                         Patch(facecolor="coral", edgecolor="black", label="CUDA")]
        plt.legend(handles=legend_elements)

        plt.tight_layout()
        plt.savefig(self.output_dir / "scatter_views_resolution.png", dpi=300)
        plt.close()

    def run_all_analyses(self):
        self.summary_statistics()
        self.correlation_analysis()
        self.anova_category_effect()
        self.plot_distributions()
        self.plot_scatter_views_vs_resolution()


def main():
    DescriptiveAnalysis().run_all_analyses()


if __name__ == "__main__":
    main()
