
from datetime import datetime

import plotly.graph_objects as go

from backend.fusion import run_fusion
from backend.orbit import get_satellites
from backend.sensors import find_next_pass


norad_id = (
    25544 if 25544 in get_satellites()
    else next(iter(get_satellites()))
)

# Use the same observation window
# for every experiment.
start_time = find_next_pass(norad_id)

scenarios = [
    ("All sensors", None, 1.0),
    ("LiDAR offline", "lidar", 1.0),
    ("Radar offline", "radar", 1.0),
    ("Optical offline", "optical", 1.0),
    ("High sensor noise", None, 5.0)
]

names = []
errors = []
uncertainties = []

print("COSMOTRACK ROBUSTNESS TEST")
print("=" * 65)

for name, dropout, noise in scenarios:

    result = run_fusion(
        norad_id=norad_id,
        dropout_sensor=dropout,
        noise_multiplier=noise,
        start_time=start_time,
        seed=42
    )

    metrics = result["metrics"]

    rmse = metrics["fused_rmse_km"] * 1000

    uncertainty = (
        metrics["final_uncertainty_km"]
        * 1000
    )

    names.append(name)
    errors.append(rmse)
    uncertainties.append(uncertainty)

    print(f"\n{name}")
    print(f"RMSE: {rmse:.2f} m")
    print(
        f"Final uncertainty: "
        f"{uncertainty:.2f} m"
    )
    print(
        f"Improvement: "
        f"{metrics['improvement_percent']:.2f}%"
    )

fig = go.Figure()

fig.add_trace(go.Bar(
    x=names,
    y=errors,
    name="Position RMSE",
    marker_color="cyan"
))

fig.add_trace(go.Bar(
    x=names,
    y=uncertainties,
    name="Final uncertainty",
    marker_color="violet"
))

fig.update_layout(
    title="COSMOTRACK | Sensor Failure Analysis",
    template="plotly_dark",
    barmode="group",
    yaxis_title="Metres",
    height=650
)

fig.show()
