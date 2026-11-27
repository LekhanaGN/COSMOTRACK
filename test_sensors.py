
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backend.sensors import simulate_sensor_series
from backend.orbit import get_satellites


norad_id = (
    25544 if 25544 in get_satellites()
    else next(iter(get_satellites()))
)

result = simulate_sensor_series(
    norad_id=norad_id,
    duration_seconds=300,
    step_seconds=10
)

data = [
    m for m in result["measurements"]
    if m["visible"]
]

if not data:
    raise RuntimeError("No visible measurements")

timestamps = [m["timestamp"] for m in data]

lidar_error = [
    (
        m["lidar"]["range_km"]
        - m["reference"]["range_km"]
    ) * 1000
    for m in data
]

radar_error = [
    (
        m["radar"]["range_km"]
        - m["reference"]["range_km"]
    ) * 1000
    for m in data
]

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "LiDAR Range Error (m)",
        "Radar Range Error (m)",
        "Radar Range Rate (km/s)",
        "Optical Azimuth & Elevation"
    )
)

# LiDAR error
fig.add_trace(
    go.Scatter(
        x=timestamps,
        y=lidar_error,
        mode="lines+markers",
        name="LiDAR error",
        line=dict(color="cyan")
    ),
    row=1, col=1
)

# Radar error
fig.add_trace(
    go.Scatter(
        x=timestamps,
        y=radar_error,
        mode="lines+markers",
        name="Radar error",
        line=dict(color="orange")
    ),
    row=1, col=2
)

# Radar range rate
fig.add_trace(
    go.Scatter(
        x=timestamps,
        y=[
            m["reference"]["range_rate_km_s"]
            for m in data
        ],
        name="Reference range rate",
        line=dict(color="white", dash="dash")
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=timestamps,
        y=[
            m["radar"]["range_rate_km_s"]
            for m in data
        ],
        name="Radar range rate",
        line=dict(color="orange")
    ),
    row=2, col=1
)

# Optical measurements
fig.add_trace(
    go.Scatter(
        x=timestamps,
        y=[
            m["optical"]["azimuth_deg"]
            for m in data
        ],
        name="Azimuth",
        line=dict(color="violet")
    ),
    row=2, col=2
)

fig.add_trace(
    go.Scatter(
        x=timestamps,
        y=[
            m["optical"]["elevation_deg"]
            for m in data
        ],
        name="Elevation",
        line=dict(color="cyan")
    ),
    row=2, col=2
)

fig.update_layout(
    title=(
        "COSMOTRACK | Simulated Sensor Analytics"
        f" | {result['name']}"
    ),
    template="plotly_dark",
    height=750,
    showlegend=True
)

fig.show()
