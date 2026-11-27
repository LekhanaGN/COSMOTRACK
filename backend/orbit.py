
from datetime import datetime, timezone, timedelta
from functools import lru_cache

import numpy as np

from skyfield.api import EarthSatellite, load, wgs84
from skyfield.framelib import itrs

from backend.catalog import load_catalog


# Use bundled time data for offline operation.
ts = load.timescale(builtin=True)


@lru_cache(maxsize=1)
def get_satellites():
    catalogue = load_catalog()

    satellites = {}

    for row in catalogue:
        satellite = EarthSatellite.from_omm(ts, row)

        norad_id = int(row["NORAD_CAT_ID"])

        satellites[norad_id] = {
            "satellite": satellite,
            "metadata": row
        }

    return satellites


def list_satellites():
    satellites = get_satellites()

    return [
        {
            "id": norad_id,
            "name": item["metadata"]["OBJECT_NAME"],
            "type": item["metadata"]["COSMOTRACK_TYPE"],
            "epoch": item["metadata"]["EPOCH"],
            "inclination_deg": float(
                item["metadata"]["INCLINATION"]
            ),
            "eccentricity": float(
                item["metadata"]["ECCENTRICITY"]
            ),
            "orbital_period_min": round(
                1440 / float(
                    item["metadata"]["MEAN_MOTION"]
                ), 2
            )
        }
        for norad_id, item in satellites.items()
    ]


def get_state(norad_id, at_time=None):
    satellites = get_satellites()

    if norad_id not in satellites:
        raise KeyError(f"Unknown object: {norad_id}")

    item = satellites[norad_id]
    satellite = item["satellite"]

    if at_time is None:
        at_time = datetime.now(timezone.utc)

    t = ts.from_datetime(at_time)

    geocentric = satellite.at(t)

    position = geocentric.position.km
    velocity = geocentric.velocity.km_per_s

    if not (
        np.isfinite(position).all()
        and np.isfinite(velocity).all()
    ):
        raise ValueError("Invalid SGP4 state")

    earth_fixed = geocentric.frame_xyz(itrs).km

    lat, lon = wgs84.latlon_of(geocentric)
    altitude = wgs84.height_of(geocentric).km

    return {
        "id": norad_id,
        "name": satellite.name,
        "type": item["metadata"]["COSMOTRACK_TYPE"],
        "timestamp": at_time.isoformat(),
        "catalog_epoch": item["metadata"]["EPOCH"],
        "epoch_age_days": round(
            abs(float(t.tt - satellite.epoch.tt)), 3
        ),
        "position_frame": "GCRS",
        "position_km": position.tolist(),
        "velocity_km_s": velocity.tolist(),
        "earth_fixed_frame": "ITRS",
        "earth_fixed_position_km": earth_fixed.tolist(),
        "latitude_deg": float(lat.degrees),
        "longitude_deg": float(lon.degrees),
        "altitude_km": float(altitude),
        "speed_km_s": float(np.linalg.norm(velocity)),
        "source": "CelesTrak",
        "method": "SGP4"
    }


def get_trajectory(
    norad_id,
    duration_minutes=100,
    step_seconds=60,
    start_time=None
):
    if start_time is None:
        start_time = datetime.now(timezone.utc)

    trajectory = []

    for seconds in range(
        0,
        duration_minutes * 60 + 1,
        step_seconds
    ):
        current_time = start_time + timedelta(
            seconds=seconds
        )

        state = get_state(norad_id, current_time)

        trajectory.append({
            "timestamp": state["timestamp"],
            "position_km": state["position_km"],
            "earth_fixed_position_km":
                state["earth_fixed_position_km"],
            "latitude_deg": state["latitude_deg"],
            "longitude_deg": state["longitude_deg"],
            "altitude_km": state["altitude_km"]
        })

    return {
        "id": norad_id,
        "name": get_satellites()[norad_id][
            "satellite"
        ].name,
        "duration_minutes": duration_minutes,
        "step_seconds": step_seconds,
        "position_frame": "GCRS",
        "earth_fixed_frame": "ITRS",
        "trajectory": trajectory
    }


if __name__ == "__main__":
    import json

    satellites = list_satellites()

    print(f"Loaded {len(satellites)} orbital objects")

    norad_id = (
        25544 if 25544 in get_satellites()
        else satellites[0]["id"]
    )

    state = get_state(norad_id)

    print("\nCURRENT ORBITAL STATE:")
    print(json.dumps(state, indent=2))

    orbit = get_trajectory(norad_id)

    print("\nTRAJECTORY PREDICTION:")
    print(f"Object: {orbit['name']}")
    print(f"Points: {len(orbit['trajectory'])}")
    print("Orbital propagation successful!")
