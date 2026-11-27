
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from backend.anomaly import (
    run_anomaly_demo,
    evaluate_model,
    NORMAL_SIGMA,
    FEATURE_NAMES,
    MODEL_NAME,
    STATISTICAL_THRESHOLD
)


demo = run_anomaly_demo()

metrics = evaluate_model()

timeline = demo["timeline"]

times = np.array([
    row["elapsed_seconds"]
    for row in timeline
])

scores = np.array([
    row["anomaly_score"]
    for row in timeline
])

ml_scores = np.array([
    row["ml_score"]
    for row in timeline
])

detected = np.array([
    row["is_anomaly"]
    for row in timeline
])

ground_truth = np.array([
    row["ground_truth_anomaly"]
    for row in timeline
])

features = np.array([
    [
        row["features"][name]
        for name in FEATURE_NAMES
    ]
    for row in timeline
])

normalised = (
    np.abs(features)
    / NORMAL_SIGMA
)

fig = make_subplots(
    rows=2,
    cols=2,

    subplot_titles=(
    "Hybrid Anomaly Score Timeline",
    "Orbital Residual Analysis",
    "Multi-Sensor Residual Analysis",
    "Hybrid Model — Confusion Matrix"
),

    horizontal_spacing=0.12,

    vertical_spacing=0.16
)


# ==========================================
# ANOMALY SCORE TIMELINE
# ==========================================

fig.add_trace(
    go.Scatter(
        x=times,
        y=scores,

        mode="lines",

        line=dict(
            color="cyan",
            width=2
        ),

        name="Hybrid Anomaly score"
    ),

    row=1,
    col=1
)
fig.add_trace(
    go.Scatter(
        x=times,
        y=ml_scores,
        mode="lines",
        line=dict(
            color="violet",
            width=1.5,
            dash="dot"
        ),
        name="Isolation Forest score"
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=times[detected],
        y=scores[detected],

        mode="markers",

        marker=dict(
            color="red",
            size=9
        ),

        text=[
            row["suspected_type"]
            for row in timeline
            if row["is_anomaly"]
        ],

        name="Detected anomaly"
    ),

    row=1,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=times[ground_truth],
        y=scores[ground_truth],

        mode="markers",

        marker=dict(
            color="yellow",
            size=13,
            symbol="circle-open"
        ),

        name="Injected anomaly"
    ),

    row=1,
    col=1
)

fig.add_hline(
    y=0,

    line_dash="dash",

    line_color="orange",

    annotation_text="Decision threshold",

    row=1,
    col=1
)


# ==========================================
# ORBITAL RESIDUAL ANALYSIS
# ==========================================

fig.add_trace(
    go.Scatter(
        x=times,

        y=normalised[:, 0],

        mode="lines",

        line=dict(
            color="cyan",
            width=2
        ),

        name="Position residual"
    ),

    row=1,
    col=2
)

fig.add_trace(
    go.Scatter(
        x=times,

        y=normalised[:, 1],

        mode="lines",

        line=dict(
            color="orange",
            width=2
        ),

        name="Velocity residual"
    ),

    row=1,
    col=2
)

fig.add_hline(
    y=STATISTICAL_THRESHOLD,
    line_dash="dash",
    line_color="red",
    annotation_text="5x statistical threshold",
    row=1,
    col=2
)

# ==========================================
# MULTI-SENSOR RESIDUAL ANALYSIS
# ==========================================

sensor_features = [
    (2, "LiDAR", "cyan"),
    (3, "Radar range", "orange"),
    (4, "Radar range rate", "lime"),
    (5, "Optical", "violet")
]

for index, name, color in sensor_features:

    fig.add_trace(
        go.Scatter(
            x=times,

            y=normalised[:, index],

            mode="lines",

            line=dict(
                color=color,
                width=2
            ),

            name=name
        ),

        row=2,
        col=1
    )

fig.add_hline(
    y=STATISTICAL_THRESHOLD,
    line_dash="dash",
    line_color="red",
    annotation_text="5x statistical threshold",
    row=2,
    col=1
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = metrics["confusion_matrix"]

matrix = [
    [
        cm["true_negative"],
        cm["false_positive"]
    ],
    [
        cm["false_negative"],
        cm["true_positive"]
    ]
]

fig.add_trace(
    go.Heatmap(
        z=matrix,

        x=[
            "Predicted Normal",
            "Predicted Anomaly"
        ],

        y=[
            "Actual Normal",
            "Actual Anomaly"
        ],

        text=matrix,

        texttemplate="%{text}",

        textfont=dict(size=22),

        colorscale="Blues",

        showscale=False
    ),

    row=2,
    col=2
)

fig.update_yaxes(
    autorange="reversed",
    row=2,
    col=2
)


# ==========================================
# AXIS LABELS
# ==========================================

for row, col in [
    (1, 1),
    (1, 2),
    (2, 1)
]:

    fig.update_xaxes(
        title_text="Elapsed time (s)",
        row=row,
        col=col
    )

fig.update_yaxes(
    title_text="Anomaly score",
    row=1,
    col=1
)

fig.update_yaxes(
    title_text="Residual / noise scale",
    row=1,
    col=2
)

fig.update_yaxes(
    title_text="Residual / noise scale",
    row=2,
    col=1
)


# ==========================================
# DASHBOARD LAYOUT
# ==========================================

fig.update_layout(
    title=(
        "COSMOTRACK | HYBRID AI ANOMALY DETECTION"
        f" | {demo['satellite_name']}"
        "<br><sup>"
        "Isolation Forest + Statistical Residual Analysis"
        "<br>"
        f"Precision: {metrics['precision']:.1%}"
        f" | Recall: {metrics['recall']:.1%}"
        f" | F1: {metrics['f1_score']:.1%}"
        f" | Accuracy: {metrics['accuracy']:.1%}"
        " | SIMULATED RESIDUALS"
        "</sup>"
    ),

    template="plotly_dark",

    height=900,

    legend=dict(
        orientation="h",
        y=-0.16
    ),

    margin=dict(
        t=140,
        b=120
    )
)

print("=" * 65)
print("COSMOTRACK | HYBRID AI PERFORMANCE")
print("=" * 65)

print("Model:", MODEL_NAME)

print(
    f"\n{'Metric':<25}"
    f"{'ML Only':>15}"
    f"{'Hybrid':>15}"
)

print("-" * 65)

for key in [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "false_positive_rate"
]:

    ml_value = metrics["ml_only_metrics"][key]

    hybrid_value = metrics[key]

    print(
        f"{key:<25}"
        f"{ml_value:>14.2%}"
        f"{hybrid_value:>14.2%}"
    )

print("\nRECALL IMPROVEMENT:")

print(
    f"{metrics['recall_improvement_percentage_points']:.2f}"
    " percentage points"
)

print(
    "\nInjected anomalies:",
    demo["injected_anomalies"]
)

print(
    "Detected anomalies:",
    demo["detected_anomalies"]
)

print("\nDETECTION SOURCES")

for source, count in demo[
    "detection_source_counts"
].items():

    print(f"{source}: {count}")

print("\nANOMALY ALERTS")

for alert in demo["alerts"][:10]:

    print(
        alert["timestamp"],
        "|",
        alert["severity"],
        "|",
        alert["suspected_type"],
        "|",
        alert["detection_source"]
    )

    print(
        "Explanation:",
        alert["explanation"]
    )

    print(
        "Recommended action:",
        alert["recommended_action"]
    )

    print("-" * 65)
# Save an interactive offline dashboard.
fig.write_html(
    "anomaly_dashboard.html",
    include_plotlyjs=True
)

fig.show()
