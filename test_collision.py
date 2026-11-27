
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from backend.collision import synthetic_encounter


result = synthetic_encounter(
    miss_distance_km=0.45,
    relative_speed_km_s=12.0,
    tca_seconds=600
)

alert = result["alert"]

trajectory = result["trajectory"]

times = np.array([
    p["seconds_from_tca"]
    for p in trajectory
])

distances = np.array([
    p["separation_km"]
    for p in trajectory
])

positions_a = np.array([
    p["position_a_km"]
    for p in trajectory
])

positions_b = np.array([
    p["position_b_km"]
    for p in trajectory
])

# Focus on the immediate encounter.
mask = np.abs(times) <= 2.0

# Centre the visualisation near TCA.
origin = np.array(
    alert["position_a_km"]
)

local_a = positions_a[mask] - origin

local_b = positions_b[mask] - origin

tca_a = (
    np.array(alert["position_a_km"])
    - origin
)

tca_b = (
    np.array(alert["position_b_km"])
    - origin
)

fig = make_subplots(
    rows=2,
    cols=2,

    specs=[
        [
            {"type": "scene"},
            {"type": "xy"}
        ],
        [
            {"type": "indicator"},
            {"type": "indicator"}
        ]
    ],

    subplot_titles=(
        "3D Close-Approach Geometry",
        "Separation vs Time",
        "Minimum Separation",
        "Relative Encounter Velocity"
    ),

    row_heights=[0.72, 0.28],

    vertical_spacing=0.15
)


# ==========================================
# 3D SATELLITE TRAJECTORY
# ==========================================

fig.add_trace(
    go.Scatter3d(
        x=local_a[:, 0],
        y=local_a[:, 1],
        z=local_a[:, 2],

        mode="lines",

        line=dict(
            color="cyan",
            width=6
        ),

        name="Satellite trajectory"
    ),

    row=1,
    col=1
)


# ==========================================
# 3D DEBRIS TRAJECTORY
# ==========================================

fig.add_trace(
    go.Scatter3d(
        x=local_b[:, 0],
        y=local_b[:, 1],
        z=local_b[:, 2],

        mode="lines",

        line=dict(
            color="orange",
            width=6
        ),

        name="Synthetic debris trajectory"
    ),

    row=1,
    col=1
)


# ==========================================
# POSITIONS AT TCA
# ==========================================

fig.add_trace(
    go.Scatter3d(
        x=[tca_a[0]],
        y=[tca_a[1]],
        z=[tca_a[2]],

        mode="markers+text",

        marker=dict(
            size=7,
            color="cyan"
        ),

        text=["Satellite at TCA"],

        textposition="top center",

        name="Satellite at TCA"
    ),

    row=1,
    col=1
)

fig.add_trace(
    go.Scatter3d(
        x=[tca_b[0]],
        y=[tca_b[1]],
        z=[tca_b[2]],

        mode="markers+text",

        marker=dict(
            size=6,
            color="orange"
        ),

        text=["Debris at TCA"],

        textposition="bottom center",

        name="Debris at TCA"
    ),

    row=1,
    col=1
)


# ==========================================
# MISS DISTANCE LINE
# ==========================================

fig.add_trace(
    go.Scatter3d(
        x=[tca_a[0], tca_b[0]],
        y=[tca_a[1], tca_b[1]],
        z=[tca_a[2], tca_b[2]],

        mode="lines",

        line=dict(
            color="red",
            width=8
        ),

        name="Miss distance"
    ),

    row=1,
    col=1
)


# ==========================================
# SEPARATION DISTANCE GRAPH
# ==========================================

fig.add_trace(
    go.Scatter(
        x=times[mask],

        y=distances[mask],

        mode="lines+markers",

        line=dict(
            color="cyan",
            width=3
        ),

        name="Separation distance"
    ),

    row=1,
    col=2
)

fig.add_hrect(
    y0=0,
    y1=1,

    fillcolor="red",

    opacity=0.15,

    line_width=0,

    row=1,
    col=2
)

fig.add_hline(
    y=1,

    line_dash="dash",

    line_color="red",

    annotation_text="Critical threshold",

    row=1,
    col=2
)

fig.add_vline(
    x=0,

    line_dash="dash",

    line_color="white",

    annotation_text="TCA",

    row=1,
    col=2
)

fig.add_trace(
    go.Scatter(
        x=[0],

        y=[alert["miss_distance_km"]],

        mode="markers",

        marker=dict(
            size=12,
            color="red"
        ),

        name="Closest approach"
    ),

    row=1,
    col=2
)


# ==========================================
# MISS DISTANCE INDICATOR
# ==========================================

fig.add_trace(
    go.Indicator(
        mode="number",

        value=(
            alert["miss_distance_km"] * 1000
        ),

        number={
            "suffix": " m",
            "valueformat": ".1f",
            "font": {
                "color": "red",
                "size": 48
            }
        }
    ),

    row=2,
    col=1
)


# ==========================================
# RELATIVE VELOCITY INDICATOR
# ==========================================

fig.add_trace(
    go.Indicator(
        mode="number",

        value=(
            alert["relative_velocity_km_s"]
        ),

        number={
            "suffix": " km/s",
            "valueformat": ".2f",
            "font": {
                "color": "orange",
                "size": 48
            }
        }
    ),

    row=2,
    col=2
)


# ==========================================
# LAYOUT
# ==========================================

fig.update_xaxes(
    title_text="Seconds from TCA",
    row=1,
    col=2
)

fig.update_yaxes(
    title_text="Separation (km)",
    row=1,
    col=2
)

fig.update_layout(
    title=(
        "COSMOTRACK | SYNTHETIC CONJUNCTION"
        f" | {alert['risk_level']} PRIORITY"
    ),

    template="plotly_dark",

    height=850,

    scene=dict(
        aspectmode="data",

        xaxis_title="Local X (km)",

        yaxis_title="Local Y (km)",

        zaxis_title="Local Z (km)"
    )
)

print("CONJUNCTION ALERT")
print("=" * 50)

print(
    "Satellite:",
    alert["object_a"]["name"]
)

print(
    "Debris:",
    alert["object_b"]["name"]
)

print("TCA:", alert["tca"])

print(
    "Miss distance:",
    alert["miss_distance_km"],
    "km"
)

print(
    "Risk:",
    alert["risk_level"]
)

print(
    "Recommended action:",
    alert["recommended_action"]
)

fig.show()
