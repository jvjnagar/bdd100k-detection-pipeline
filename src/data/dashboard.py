import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from .analysis import (
    annotations_per_image_stats,
    bbox_size_stats,
    class_distribution,
    co_occurrence_matrix,
    images_per_class,
    split_comparison,
)
from .anomalies import (
    detect_class_imbalance,
    detect_tiny_boxes,
    find_outlier_images,
)
from .parser import DETECTION_CLASSES, ImageAnnotation, load_split

DATA_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "/app/data")

SIZE_ORDER = ["tiny", "small", "medium", "large"]

def _instances_vs_images(annotations: List[ImageAnnotation]) -> pd.DataFrame:
    """Combine per-class instance counts with the number of unique images.

    Args:
        annotations: List of image annotations.

    Returns:
        DataFrame with columns class, instances, images and avg_per_image.
    """
    dist = class_distribution(annotations)[["class", "count"]].rename(
        columns={"count": "instances"}
    )
    img_counts = images_per_class(annotations)
    dist["images"] = dist["class"].map(img_counts).fillna(0).astype(int)
    dist["avg_per_image"] = np.where(
        dist["images"] > 0,
        (dist["instances"] / dist["images"].replace(0, np.nan)).round(2),
        0.0,
    )
    return dist


def _size_category_counts(size_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-box size DataFrame into per-class size-category counts.

    Args:
        size_df: Output of bbox_size_stats.

    Returns:
        DataFrame with columns class, size_category and count.
    """
    if size_df.empty:
        return pd.DataFrame(columns=["class", "size_category", "count"])
    return (
        size_df.groupby(["class", "size_category"]).size().reset_index(name="count")
    )


def _pct_tiny_per_class(size_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the percentage of tiny (<32x32 px) boxes for each class.

    Args:
        size_df: Output of bbox_size_stats.

    Returns:
        DataFrame with columns class and pct_tiny (descending).
    """
    if size_df.empty:
        return pd.DataFrame(columns=["class", "pct_tiny"])
    df = size_df.copy()
    df["is_tiny"] = (df["size_category"] == "tiny").astype(int)
    pct = (
        df.groupby("class")["is_tiny"].mean().mul(100).round(1).reset_index(
            name="pct_tiny"
        )
    )
    return pct.sort_values("pct_tiny", ascending=False).reset_index(drop=True)


def _per_class_summary(
    train_anns: List[ImageAnnotation],
    val_anns: List[ImageAnnotation],
    size_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a per-class summary table combining distribution + anomaly metrics.

    Args:
        train_anns: Training annotations.
        val_anns: Validation annotations.
        size_df: Output of bbox_size_stats for the training split.

    Returns:
        Summary DataFrame sorted by training instance count (descending).
    """
    comp = split_comparison(train_anns, val_anns)
    imb = detect_class_imbalance(train_anns)[["class", "imbalance_ratio"]]
    pct_tiny = _pct_tiny_per_class(size_df)
    img_counts = images_per_class(train_anns)

    if size_df.empty:
        med_area = pd.DataFrame({"class": DETECTION_CLASSES, "median_area": 0.0})
    else:
        med_area = (
            size_df.groupby("class")["area"].median().round(0).reset_index(
                name="median_area"
            )
        )

    summary = (
        comp.merge(imb, on="class", how="left")
        .merge(pct_tiny, on="class", how="left")
        .merge(med_area, on="class", how="left")
    )
    summary["images"] = summary["class"].map(img_counts).fillna(0).astype(int)
    summary["pct_tiny"] = summary["pct_tiny"].fillna(0.0)
    summary["median_area"] = summary["median_area"].fillna(0.0)

    column_order = [
        "class", "train_count", "val_count", "train_pct", "val_pct",
        "ratio_diff", "imbalance_ratio", "images", "pct_tiny", "median_area",
    ]
    summary = summary[column_order].sort_values(
        "train_count", ascending=False
    ).reset_index(drop=True)
    return summary


def _key_findings(summary: pd.DataFrame) -> List[str]:
    """Auto-generate plain-language findings about anomalies/patterns.

    Args:
        summary: Output of _per_class_summary.

    Returns:
        List of human-readable finding strings.
    """
    if summary.empty or summary["train_count"].sum() == 0:
        return ["No annotations available to analyse."]

    s = summary.copy()
    s["imbalance_ratio"] = s["imbalance_ratio"].replace([np.inf], np.nan)
    findings: List[str] = []

    dom = s.loc[s["train_count"].idxmax()]
    findings.append(
        f"Dominant class: '{dom['class']}' accounts for "
        f"{dom['train_pct']:.1f}% of all training instances."
    )
    if s["imbalance_ratio"].notna().any():
        imb = s.loc[s["imbalance_ratio"].idxmax()]
        findings.append(
            f"Most under-represented class: '{imb['class']}' "
            f"(~{imb['imbalance_ratio']:.0f}:1 vs. the most common class)."
        )
    if s["pct_tiny"].notna().any() and s["pct_tiny"].max() > 0:
        tin = s.loc[s["pct_tiny"].idxmax()]
        findings.append(
            f"Smallest objects: '{tin['class']}' has {tin['pct_tiny']:.0f}% of "
            f"its boxes under 32x32 px - the hardest to detect."
        )
    if s["ratio_diff"].notna().any():
        shift = s.loc[s["ratio_diff"].idxmax()]
        findings.append(
            f"Largest train/val gap: '{shift['class']}' differs by "
            f"{shift['ratio_diff']:.2f} percentage points between splits."
        )
    return findings


def _per_class_detail_fig(
    selected_class: str,
    size_df: pd.DataFrame,
    train_counts: Dict[str, int],
    val_counts: Dict[str, int],
) -> go.Figure:
    """Build the interactive per-class size-distribution figure.

    Args:
        selected_class: Class to visualise.
        size_df: Output of bbox_size_stats for the training split.
        train_counts: Mapping class -> training instance count.
        val_counts: Mapping class -> validation instance count.

    Returns:
        A Plotly bar figure of the selected class size-category distribution.
    """
    if not size_df.empty:
        sub = size_df[size_df["class"] == selected_class]
    else:
        sub = pd.DataFrame(columns=["size_category"])

    if not sub.empty:
        counts = sub["size_category"].value_counts().reindex(
            SIZE_ORDER, fill_value=0
        )
    else:
        counts = pd.Series([0, 0, 0, 0], index=SIZE_ORDER)

    total = int(counts.sum())
    pct_tiny = round(counts["tiny"] / total * 100, 1) if total else 0.0
    t_count = int(train_counts.get(selected_class, 0))
    v_count = int(val_counts.get(selected_class, 0))

    fig = px.bar(
        x=SIZE_ORDER,
        y=counts.values,
        color=SIZE_ORDER,
        labels={"x": "Size category", "y": "Box count"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        showlegend=False,
        title=(
            f"'{selected_class}' - {t_count:,} train / {v_count:,} val instances "
            f"- {pct_tiny}% tiny boxes (training size distribution)"
        ),
    )
    return fig


def _summary_table_fig(summary: pd.DataFrame) -> go.Figure:
    """Render the per-class summary DataFrame as a Plotly table figure."""
    disp = summary.copy()
    disp["imbalance_ratio"] = disp["imbalance_ratio"].replace([np.inf], np.nan)
    headers = [
        "Class", "Train #", "Val #", "Train %", "Val %", "Diff %",
        "Imbalance", "# Images", "% Tiny", "Median Area",
    ]
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=headers,
                    fill_color="#2c3e50",
                    font=dict(color="white", size=12),
                    align="left",
                ),
                cells=dict(
                    values=[
                        disp["class"],
                        disp["train_count"],
                        disp["val_count"],
                        disp["train_pct"],
                        disp["val_pct"],
                        disp["ratio_diff"],
                        disp["imbalance_ratio"].round(1),
                        disp["images"],
                        disp["pct_tiny"],
                        disp["median_area"],
                    ],
                    align="left",
                    fill_color="#f7f9fb",
                ),
            )
        ]
    )
    fig.update_layout(
        title="Per-Class Summary (training distribution + anomaly metrics)",
        margin=dict(t=40, b=10),
    )
    return fig


def _stat_card(title: str, value: str, subtitle: str = "") -> html.Div:
    """Create a small styled metric card."""
    return html.Div(
        [
            html.H4(title, style={"margin": "0 0 6px 0", "color": "#2c3e50"}),
            html.P(
                value,
                style={"fontSize": "26px", "fontWeight": "bold", "margin": "0"},
            ),
            html.P(subtitle, style={"margin": "4px 0 0 0", "color": "#7f8c8d"}),
        ],
        style={
            "flex": "1",
            "minWidth": "180px",
            "padding": "16px",
            "margin": "8px",
            "background": "#ffffff",
            "border": "1px solid #e1e4e8",
            "borderRadius": "8px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
        },
    )

def create_dashboard(
    train_anns: List[ImageAnnotation], val_anns: List[ImageAnnotation]
) -> Dash:
    """Create and configure the Dash application.

    Args:
        train_anns: Training set annotations.
        val_anns: Validation set annotations.

    Returns:
        Configured Dash application instance.
    """
    app = Dash(__name__)

    # Core statistics (computed once)
    train_dist = class_distribution(train_anns)
    val_dist = class_distribution(val_anns)
    comparison = split_comparison(train_anns, val_anns)
    train_size_df = bbox_size_stats(train_anns)
    co_matrix = co_occurrence_matrix(train_anns)
    train_img_stats = annotations_per_image_stats(train_anns)
    val_img_stats = annotations_per_image_stats(val_anns)

    # Derived analysis (distribution + anomalies/patterns)
    ivi_df = _instances_vs_images(train_anns)
    imbalance_df = detect_class_imbalance(train_anns)
    size_cat_df = _size_category_counts(train_size_df)
    pct_tiny_df = _pct_tiny_per_class(train_size_df)
    summary_df = _per_class_summary(train_anns, val_anns, train_size_df)
    findings = _key_findings(summary_df)

    tiny_boxes = detect_tiny_boxes(train_anns)
    tiny_total = len(tiny_boxes)
    tiny_per_class = Counter(cat for _, cat, _ in tiny_boxes)
    worst_tiny_cls = (
        tiny_per_class.most_common(1)[0][0] if tiny_per_class else "-"
    )
    outliers = find_outlier_images(train_anns)
    high_out = len(outliers["high"])

    train_counts = dict(zip(train_dist["class"], train_dist["count"]))
    val_counts = dict(zip(val_dist["class"], val_dist["count"]))
    default_class = (
        str(train_dist.iloc[0]["class"])
        if not train_dist.empty
        else DETECTION_CLASSES[0]
    )

    # Figures
    fig_train_dist = px.bar(
        train_dist, x="class", y="count", color="class", text="count",
        title="Training Set: Class Distribution (instances per class)",
    )
    fig_val_dist = px.bar(
        val_dist, x="class", y="count", color="class", text="count",
        title="Validation Set: Class Distribution (instances per class)",
    )

    fig_ivi = go.Figure()
    fig_ivi.add_trace(
        go.Bar(name="Instances (boxes)", x=ivi_df["class"], y=ivi_df["instances"])
    )
    fig_ivi.add_trace(
        go.Bar(
            name="Images containing class", x=ivi_df["class"], y=ivi_df["images"]
        )
    )
    fig_ivi.update_layout(
        barmode="group",
        title="Instances vs. Unique Images per Class (Training)",
    )

    fig_comparison = go.Figure()
    fig_comparison.add_trace(
        go.Bar(name="Train %", x=comparison["class"], y=comparison["train_pct"])
    )
    fig_comparison.add_trace(
        go.Bar(name="Val %", x=comparison["class"], y=comparison["val_pct"])
    )
    fig_comparison.update_layout(
        barmode="group", title="Train vs Val: Class Proportions (%)"
    )

    fig_shift = px.bar(
        comparison.sort_values("ratio_diff", ascending=False),
        x="class", y="ratio_diff", text="ratio_diff",
        color="ratio_diff", color_continuous_scale="Blues",
        title="Train vs Val Distribution Shift abs(train% - val%) (split anomaly)",
    )

    imb_plot = imbalance_df.replace([np.inf], np.nan).dropna(
        subset=["imbalance_ratio"]
    )
    fig_imbalance = px.bar(
        imb_plot, x="class", y="imbalance_ratio", text="imbalance_ratio",
        color="imbalance_ratio", color_continuous_scale="OrRd", log_y=True,
        title="Class Imbalance Ratio (max_count / class_count) - log scale",
    )

    if not size_cat_df.empty:
        fig_size_cat = px.bar(
            size_cat_df, x="class", y="count", color="size_category",
            category_orders={"size_category": SIZE_ORDER}, barmode="stack",
            title="Bounding-Box Size Categories per Class (Training)",
        )
    else:
        fig_size_cat = go.Figure()

    fig_pct_tiny = px.bar(
        pct_tiny_df, x="class", y="pct_tiny", text="pct_tiny",
        color="pct_tiny", color_continuous_scale="Reds",
        title="Percent Tiny Boxes (<32x32 px) per Class - detection challenge",
    )

    fig_summary_table = _summary_table_fig(summary_df)

    fig_heatmap = px.imshow(
        co_matrix, text_auto=True, color_continuous_scale="YlOrRd",
        title="Class Co-occurrence Matrix (images containing both classes)",
    )

    if len(train_size_df) > 0:
        fig_sizes = px.box(
            train_size_df, x="class", y="area", log_y=True,
            title="Bounding Box Area by Class (Training)",
        )
        fig_aspect = px.box(
            train_size_df[train_size_df["aspect_ratio"] < 5], x="class",
            y="aspect_ratio", title="Aspect Ratio by Class (Training)",
        )
    else:
        fig_sizes = go.Figure()
        fig_aspect = go.Figure()

    # Layout
    app.layout = html.Div(
        [
            html.H1("BDD100K Object Detection - Dataset Analysis Dashboard"),
            html.P(
                "Distribution, train/val split, and per-class anomaly and "
                "pattern analysis for the 10 detection classes.",
                style={"color": "#7f8c8d"},
            ),
            html.Hr(),
            html.H2("Key Findings"),
            html.Ul([html.Li(f) for f in findings]),
            html.Hr(),
            html.H2("Dataset Overview"),
            html.Div(
                [
                    _stat_card(
                        "Training images",
                        f"{train_img_stats['total_images']:,}",
                        f"avg {train_img_stats['mean']} boxes/image",
                    ),
                    _stat_card(
                        "Validation images",
                        f"{val_img_stats['total_images']:,}",
                        f"avg {val_img_stats['mean']} boxes/image",
                    ),
                    _stat_card(
                        "Tiny-box candidates",
                        f"{tiny_total:,}",
                        f"side <16 px - worst: {worst_tiny_cls}",
                    ),
                    _stat_card(
                        "Count outlier images",
                        f"{high_out:,}",
                        "z-score > 3 (unusually crowded)",
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap"},
            ),
            html.Hr(),
            html.H2("1 - Class Distribution"),
            dcc.Graph(figure=fig_train_dist),
            dcc.Graph(figure=fig_val_dist),
            dcc.Graph(figure=fig_ivi),
            html.Hr(),
            html.H2("2 - Train / Val Split Analysis"),
            dcc.Graph(figure=fig_comparison),
            dcc.Graph(figure=fig_shift),
            html.Hr(),
            html.H2("3 - Anomalies and Patterns per Class"),
            dcc.Graph(figure=fig_imbalance),
            dcc.Graph(figure=fig_size_cat),
            dcc.Graph(figure=fig_pct_tiny),
            html.H3("Interactive per-class deep-dive"),
            html.P("Select a class to inspect its bounding-box size pattern:"),
            dcc.Dropdown(
                id="class-selector",
                options=[{"label": c, "value": c} for c in DETECTION_CLASSES],
                value=default_class,
                clearable=False,
                style={"width": "320px", "marginBottom": "12px"},
            ),
            dcc.Graph(
                id="per-class-detail",
                figure=_per_class_detail_fig(
                    default_class, train_size_df, train_counts, val_counts
                ),
            ),
            html.H3("Per-class summary table"),
            dcc.Graph(figure=fig_summary_table),
            html.Hr(),
            html.H2("Supporting Detail"),
            dcc.Graph(figure=fig_sizes),
            dcc.Graph(figure=fig_aspect),
            dcc.Graph(figure=fig_heatmap),
        ],
        style={"padding": "20px", "maxWidth": "1200px", "margin": "0 auto"},
    )

    # Callback: interactive per-class deep-dive
    @app.callback(
        Output("per-class-detail", "figure"),
        Input("class-selector", "value"),
    )
    def _update_per_class(selected_class: str) -> go.Figure:
        """Update the per-class detail figure when the dropdown changes."""
        return _per_class_detail_fig(
            selected_class, train_size_df, train_counts, val_counts
        )

    return app


def main() -> None:
    """Load data and launch the dashboard."""
    data_dir = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR)

    train_anns = load_split(data_dir, "train")
    val_anns = load_split(data_dir, "val")

    if train_anns is None or val_anns is None:
        print("Error: Could not load dataset. Check data directory.")
        sys.exit(1)

    app = create_dashboard(train_anns, val_anns)
    print("Dashboard ready — open http://localhost:8050 in your browser")
    app.run(host="0.0.0.0", port=8050, debug=False)


if __name__ == "__main__":
    main()
