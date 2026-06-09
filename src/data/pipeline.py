import json
import os
import sys
from pathlib import Path

from .analysis import (
    annotations_per_image_stats,
    bbox_size_stats,
    class_distribution,
    co_occurrence_matrix,
    split_comparison,
)
from .anomalies import (
    detect_class_imbalance,
    detect_tiny_boxes,
    find_interesting_samples,
    find_outlier_images,
)
from .parser import load_split
from .visualization import (
    plot_annotations_histogram,
    plot_bbox_sizes,
    plot_class_distribution,
    plot_co_occurrence,
    plot_split_comparison,
)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/app/output"))


def run_analysis(data_dir: str, output_dir: str) -> None:
    """Execute the complete BDD100K analysis pipeline.

    Args:
        data_dir: Path to root data directory.
        output_dir: Path to output directory for results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BDD100K Object Detection Dataset Analysis")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading dataset...")
    train_anns = load_split(data_dir, "train")
    val_anns = load_split(data_dir, "val")

    if train_anns is None or val_anns is None:
        print("ERROR: Could not load dataset labels.")
        print(f"Please ensure labels are in: {data_dir}")
        sys.exit(1)

    print(f"  Train: {len(train_anns):,} images")
    print(f"  Val:   {len(val_anns):,} images")

    # Class distribution
    print("\n[2/6] Computing class distributions...")
    train_dist = class_distribution(train_anns)
    val_dist = class_distribution(val_anns)
    print("\nTraining Set Distribution:")
    print(train_dist.to_string(index=False))

    train_dist.to_csv(output_path / "train_class_distribution.csv", index=False)
    val_dist.to_csv(output_path / "val_class_distribution.csv", index=False)

    # Split comparison
    print("\n[3/6] Comparing train/val splits...")
    comparison = split_comparison(train_anns, val_anns)
    print(comparison.to_string(index=False))
    comparison.to_csv(output_path / "split_comparison.csv", index=False)

    # Bbox statistics
    print("\n[4/6] Computing bounding box statistics...")
    train_size_df = bbox_size_stats(train_anns)
    train_size_df.to_csv(output_path / "train_bbox_stats.csv", index=False)

    train_img_stats = annotations_per_image_stats(train_anns)
    val_img_stats = annotations_per_image_stats(val_anns)
    print(f"\n  Train annotations/image: {train_img_stats}")
    print(f"  Val annotations/image:   {val_img_stats}")

    # Co-occurrence
    co_matrix = co_occurrence_matrix(train_anns)
    co_matrix.to_csv(output_path / "co_occurrence_matrix.csv")

    # Anomaly detection
    print("\n[5/6] Detecting anomalies and patterns...")
    imbalance = detect_class_imbalance(train_anns)
    print("\nClass Imbalance Ratios:")
    print(imbalance.to_string(index=False))
    imbalance.to_csv(output_path / "class_imbalance.csv", index=False)

    outliers = find_outlier_images(train_anns)
    print(f"\n  High-count outlier images: {len(outliers['high'])}")
    print(f"  Low-count outlier images:  {len(outliers['low'])}")

    tiny_boxes = detect_tiny_boxes(train_anns)
    print(f"  Suspiciously tiny boxes:   {len(tiny_boxes)}")

    interesting = find_interesting_samples(train_anns)
    with open(output_path / "interesting_samples.json", "w", encoding="utf-8") as f:
        json.dump(interesting, f, indent=2)

    # Visualization
    print("\n[6/6] Generating visualizations...")
    plot_class_distribution(
        train_dist, "Training Set: Class Distribution", str(output_path / "train_distribution.png")
    )
    plot_class_distribution(
        val_dist, "Validation Set: Class Distribution", str(output_path / "val_distribution.png")
    )
    plot_split_comparison(comparison, str(output_path / "split_comparison.png"))
    plot_bbox_sizes(train_size_df, str(output_path / "bbox_sizes.png"))
    plot_co_occurrence(co_matrix, str(output_path / "co_occurrence.png"))
    plot_annotations_histogram(train_anns, str(output_path / "annotations_histogram.png"))

    # Summary report
    report = {
        "dataset": "BDD100K",
        "task": "Object Detection",
        "classes": 10,
        "train_images": len(train_anns),
        "val_images": len(val_anns),
        "train_stats": train_img_stats,
        "val_stats": val_img_stats,
        "class_imbalance": imbalance.to_dict(orient="records"),
        "outlier_images_high": len(outliers["high"]),
        "outlier_images_low": len(outliers["low"]),
        "tiny_boxes_count": len(tiny_boxes),
    }
    with open(output_path / "analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("Analysis complete! Results saved to:", output_dir)
    print("=" * 60)


def main():
    """Main entry point."""
    data_dir = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR)
    output_dir = sys.argv[2] if len(sys.argv) > 2 else str(OUTPUT_DIR)
    run_analysis(data_dir, output_dir)


if __name__ == "__main__":
    main()
