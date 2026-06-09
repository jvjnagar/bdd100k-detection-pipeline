from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .parser import DETECTION_CLASSES, ImageAnnotation


def setup_style() -> None:
    """Configure matplotlib style for consistent plots."""
    sns.set_theme(style="whitegrid", palette="husl")
    plt.rcParams.update({"figure.figsize": (12, 6), "font.size": 11})


def plot_class_distribution(
    df: pd.DataFrame, title: str, output_path: str
) -> None:
    """Plot bar chart of class distribution.

    Args:
        df: DataFrame with 'class' and 'count' columns.
        title: Plot title.
        output_path: Path to save the figure.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(df["class"], df["count"], color=sns.color_palette("husl", len(df)))
    ax.set_xlabel("Object Class")
    ax.set_ylabel("Number of Instances")
    ax.set_title(title)
    ax.set_xticklabels(df["class"], rotation=45, ha="right")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_split_comparison(
    df: pd.DataFrame, output_path: str
) -> None:
    """Plot grouped bar chart comparing train and val distributions.

    Args:
        df: DataFrame from split_comparison function.
        output_path: Path to save the figure.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Absolute counts
    x = np.arange(len(df))
    width = 0.35
    axes[0].bar(x - width / 2, df["train_count"], width, label="Train")
    axes[0].bar(x + width / 2, df["val_count"], width, label="Val")
    axes[0].set_xlabel("Object Class")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Instance Count: Train vs Val")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["class"], rotation=45, ha="right")
    axes[0].legend()

    # Percentage comparison
    axes[1].bar(x - width / 2, df["train_pct"], width, label="Train %")
    axes[1].bar(x + width / 2, df["val_pct"], width, label="Val %")
    axes[1].set_xlabel("Object Class")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_title("Class Proportion: Train vs Val")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["class"], rotation=45, ha="right")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_bbox_sizes(
    size_df: pd.DataFrame, output_path: str
) -> None:
    """Plot bounding box size distributions per class.

    Args:
        size_df: DataFrame from bbox_size_stats function.
        output_path: Path to save the figure.
    """
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Area distribution (log scale)
    sns.boxplot(data=size_df, x="class", y="area", ax=axes[0, 0])
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Bounding Box Area Distribution (log scale)")
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha="right")

    # Aspect ratio distribution
    sns.boxplot(data=size_df, x="class", y="aspect_ratio", ax=axes[0, 1])
    axes[0, 1].set_title("Aspect Ratio Distribution")
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha="right")
    axes[0, 1].set_ylim(0, 5)

    # Size category stacked bar
    size_pivot = (
        size_df.groupby(["class", "size_category"]).size().unstack(fill_value=0)
    )
    size_pivot_pct = size_pivot.div(size_pivot.sum(axis=1), axis=0) * 100
    size_pivot_pct.plot(kind="bar", stacked=True, ax=axes[1, 0])
    axes[1, 0].set_title("Size Category Distribution (%)")
    axes[1, 0].set_ylabel("Percentage")
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha="right")
    axes[1, 0].legend(title="Size", bbox_to_anchor=(1.05, 1))

    # Width vs Height scatter (sampled)
    sample = size_df.sample(min(5000, len(size_df)), random_state=42)
    sns.scatterplot(
        data=sample, x="width", y="height", hue="class", alpha=0.4, ax=axes[1, 1]
    )
    axes[1, 1].set_title("Width vs Height (sampled)")
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_co_occurrence(
    co_matrix: pd.DataFrame, output_path: str
) -> None:
    """Plot co-occurrence heatmap.

    Args:
        co_matrix: Co-occurrence DataFrame.
        output_path: Path to save the figure.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        co_matrix,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Class Co-occurrence Matrix (number of images)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_annotations_histogram(
    annotations: List[ImageAnnotation], output_path: str
) -> None:
    """Plot histogram of number of annotations per image.

    Args:
        annotations: List of image annotations.
        output_path: Path to save the figure.
    """
    setup_style()
    counts = [len(ann.bboxes) for ann in annotations]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(counts, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Number of Bounding Boxes per Image")
    ax.set_ylabel("Number of Images")
    ax.set_title("Distribution of Annotations per Image")
    ax.axvline(np.mean(counts), color="red", linestyle="--", label=f"Mean: {np.mean(counts):.1f}")
    ax.axvline(np.median(counts), color="green", linestyle="--", label=f"Median: {np.median(counts):.1f}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
