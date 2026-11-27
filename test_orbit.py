
import numpy as np
import plotly.graph_objects as go

from backend.orbit import (
    get_state,
    get_trajectory,
    get_satellites
)

norad_id = 25544

if norad_id not in get_satellites():
    norad_id = next(iter(get_satellites()))

state = get_state(norad_id)
orbit = get_trajectory(norad_id, 100, 60)

# Earth's approximate spherical surface.
radius = 6371

u = np.linspace(0, 2 * np.pi, 80)
v = np.linspace(0, np.pi, 40)

x = radius * np.outer(np.cos(u), np.sin(v))
y = radius * np.outer(np.sin(u), np.sin(v))
z = radius * np.outer(
    np.ones(np.size(u)), np.cos(v)
)

positions = np.array([
    point["position_km"]
    for point in orbit["trajectory"]
])

fig = go.Figure()

# Earth
fig.add_trace(go.Surface(
    x=x,
    y=y,
    z=z,
    colorscale="Blues",
    showscale=False,
    opacity=0.95,
    name="Earth",
    hoverinfo="skip"
))

# Predicted orbit
fig.add_trace(go.Scatter3d(
    x=positions[:, 0],
    y=positions[:, 1],
    z=positions[:, 2],
    mode="lines",
    line=dict(color="cyan", width=5),
    name="Predicted trajectory"
))

# Initial satellite position
fig.add_trace(go.Scatter3d(
    x=[positions[0, 0]],
    y=[positions[0, 1]],
    z=[positions[0, 2]],
    mode="markers+text",
    marker=dict(size=6, color="orange"),
    text=[state["name"]],
    textposition="top center",
    name="Satellite"
))

fig.update_layout(
    title="COSMOTRACK | Orbital Tracking",
    template="plotly_dark",
    height=750,
    scene=dict(
        aspectmode="data",
        xaxis_title="X (km)",
        yaxis_title="Y (km)",
        zaxis_title="Z (km)"
    )
)

fig.show()
