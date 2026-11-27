
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from backend.fusion import run_fusion
from backend.orbit import get_satellites


norad_id = (
    25544 if 25544 in get_satellites()
    else next(iter(get_satellites()))
)

result = run_fusion(
    norad_id=norad_id,
    duration_seconds=300,
    step_seconds=10
)

data = result["results"]
metrics = result["metrics"]

time = [
    r["elapsed_seconds"]
    for r in data
]

baseline_error = [
    r["baseline_error_km"] * 1000
    for r in data
]

fused_error = [
    r["fused_error_km"] * 1000
    for r in data
]

uncertainty = [
    r["position_uncertainty_km"] * 1000
    for r in data
]

baseline_velocity = [
    r["baseline_velocity_error_km_s"] * 1000
    for r in data
]

fused_velocity = [
    r["fused_velocity_error_km_s"] * 1000
    for r in data
]

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Position Tracking Error",
        "Estimated Position Uncertainty",
        "Velocity Tracking Error",
        "Position RMSE Comparison"
    )
)

# ----------------------------------
# Position tracking error
# ----------------------------------

fig.add_trace(
    go.Scatter(
        x=time,
        y=baseline_error,
        mode="lines",
        name="Without fusion",
        line=dict(
            color="orange",
            width=2
        )
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=time,
        y=fused_error,
        mode="lines+markers",
        name="With EKF",
        line=dict(
            color="cyan",
            width=2
        )
    ),
    row=1, col=1
)

# ----------------------------------
# Position uncertainty
# ----------------------------------

fig.add_trace(
    go.Scatter(
        x=time,
        y=uncertainty,
        mode="lines",
        fill="tozeroy",
        name="Position uncertainty",
        line=dict(color="violet")
    ),
    row=1, col=2
)

# ----------------------------------
# Velocity tracking error
# ----------------------------------

fig.add_trace(
    go.Scatter(
        x=time,
        y=baseline_velocity,
        name="Baseline velocity error",
        line=dict(color="orange")
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=time,
        y=fused_velocity,
        name="EKF velocity error",
        line=dict(color="cyan")
    ),
    row=2, col=1
)

# ----------------------------------
# RMSE comparison
# ----------------------------------

fig.add_trace(
    go.Bar(
        x=["Without Fusion", "With EKF"],
        y=[
            metrics["baseline_rmse_km"] * 1000,
            metrics["fused_rmse_km"] * 1000
        ],
        marker_color=[
            "orange",
            "cyan"
        ],
        text=[
            round(
                metrics["baseline_rmse_km"]
                * 1000, 2
            ),
            round(
                metrics["fused_rmse_km"]
                * 1000, 2
            )
        ],
        textposition="outside",
        name="RMSE"
    ),
    row=2, col=2
)

fig.update_yaxes(
    title_text="Error (m)",
    row=1, col=1
)

fig.update_yaxes(
    title_text="Uncertainty (m)",
    row=1, col=2
)

fig.update_yaxes(
    title_text="Error (m/s)",
    row=2, col=1
)

fig.update_yaxes(
    title_text="RMSE (m)",
    row=2, col=2
)

fig.update_xaxes(
    title_text="Elapsed time (s)",
    row=1, col=1
)

fig.update_xaxes(
    title_text="Elapsed time (s)",
    row=1, col=2
)

fig.update_xaxes(
    title_text="Elapsed time (s)",
    row=2, col=1
)

improvement = metrics[
    "improvement_percent"
]

fig.update_layout(
    title=(
        "COSMOTRACK | EKF Sensor Fusion"
        f" | {result['name']}"
        f" | RMSE improvement: "
        f"{improvement:.2f}%"
    ),
    template="plotly_dark",
    height=800
)

print("FUSION RESULTS")
print("=" * 50)

for key, value in metrics.items():
    print(f"{key}: {value:.6f}")

fig.show()
