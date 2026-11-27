
from datetime import datetime, timezone, timedelta

import numpy as np
from skyfield.api import wgs84

from backend.orbit import (
    get_satellites,
    get_state,
    ts
)


# ---------------------------------------
# Ground station configuration
# ---------------------------------------

STATION = {
    "name": "Bengaluru Observation Station",
    "latitude_deg": 12.9716,
    "longitude_deg": 77.5946,
    "elevation_m": 920.0,
    "minimum_elevation_deg": 10.0
}

GROUND_STATION = wgs84.latlon(
    latitude_degrees=STATION["latitude_deg"],
    longitude_degrees=STATION["longitude_deg"],
    elevation_m=STATION["elevation_m"]
)


# ---------------------------------------
# Sensor noise assumptions (1 sigma)
# ---------------------------------------

SENSOR_NOISE = {
    "lidar_range_km": 0.01,
    "radar_range_km": 0.1,
    "radar_range_rate_km_s": 0.01,
    "optical_azimuth_deg": 0.02,
    "optical_elevation_deg": 0.02
}


# ---------------------------------------
# Reference measurement geometry
# ---------------------------------------

def get_reference_geometry(norad_id, at_time=None):

    if at_time is None:
        at_time = datetime.now(timezone.utc)

    satellites = get_satellites()

    if norad_id not in satellites:
        raise KeyError(f"Unknown object: {norad_id}")

    satellite = satellites[norad_id]["satellite"]

    t = ts.from_datetime(at_time)

    # Satellite relative to the ground station.
    topocentric = (satellite - GROUND_STATION).at(t)

    elevation, azimuth, distance = topocentric.altaz()

    relative_position = topocentric.position.km
    relative_velocity = topocentric.velocity.km_per_s

    if not (
        np.isfinite(relative_position).all()
        and np.isfinite(relative_velocity).all()
    ):
        raise ValueError("Invalid orbital geometry")

    # Slant range.
    range_km = np.linalg.norm(relative_position)

    # Radial velocity relative to the station.
    range_rate = np.dot(
        relative_position,
        relative_velocity
    ) / range_km

    station_state = GROUND_STATION.at(t)

    visible = (
        elevation.degrees
        >= STATION["minimum_elevation_deg"]
    )

    return {
        "id": norad_id,
        "timestamp": at_time.isoformat(),
        "range_km": float(range_km),
        "range_rate_km_s": float(range_rate),
        "azimuth_deg": float(azimuth.degrees),
        "elevation_deg": float(elevation.degrees),
        "visible": bool(visible),
        "station_position_km":
            station_state.position.km.tolist(),
        "station_velocity_km_s":
            station_state.velocity.km_per_s.tolist(),
        "position_frame": "GCRS"
    }


# ---------------------------------------
# Simulate individual sensor readings
# ---------------------------------------

def simulate_measurements(
    norad_id,
    at_time=None,
    rng=None,
    noise_multiplier=1.0,
    dropout_sensor=None
):

    if rng is None:
        rng = np.random.default_rng(42)

    reference = get_reference_geometry(
        norad_id,
        at_time
    )

    def noisy(value, key):
        sigma = (
            SENSOR_NOISE[key] * noise_multiplier
        )

        return float(
            value + rng.normal(0, sigma)
        )

    # No ground-based observation below
    # the configured elevation threshold.
    if not reference["visible"]:
        return {
            "id": norad_id,
            "timestamp": reference["timestamp"],
            "visible": False,
            "reference": reference,
            "lidar": None,
            "radar": None,
            "optical": None,
            "simulated": True
        }

    lidar = {
        "range_km": noisy(
            reference["range_km"],
            "lidar_range_km"
        ),
        "sigma_range_km":
            SENSOR_NOISE["lidar_range_km"]
            * noise_multiplier
    }

    radar = {
        "range_km": noisy(
            reference["range_km"],
            "radar_range_km"
        ),
        "range_rate_km_s": noisy(
            reference["range_rate_km_s"],
            "radar_range_rate_km_s"
        ),
        "sigma_range_km":
            SENSOR_NOISE["radar_range_km"]
            * noise_multiplier,
        "sigma_range_rate_km_s":
            SENSOR_NOISE["radar_range_rate_km_s"]
            * noise_multiplier
    }

    optical = {
        "azimuth_deg": noisy(
            reference["azimuth_deg"],
            "optical_azimuth_deg"
        ) % 360,
        "elevation_deg": noisy(
            reference["elevation_deg"],
            "optical_elevation_deg"
        ),
        "sigma_azimuth_deg":
            SENSOR_NOISE["optical_azimuth_deg"]
            * noise_multiplier,
        "sigma_elevation_deg":
            SENSOR_NOISE["optical_elevation_deg"]
            * noise_multiplier
    }

    # Optional sensor dropout demonstration.
    if dropout_sensor == "lidar":
        lidar = None

    elif dropout_sensor == "radar":
        radar = None

    elif dropout_sensor == "optical":
        optical = None

    return {
        "id": norad_id,
        "timestamp": reference["timestamp"],
        "visible": True,
        "reference": reference,
        "lidar": lidar,
        "radar": radar,
        "optical": optical,
        "simulated": True
    }


# ---------------------------------------
# Find next visible satellite pass
# ---------------------------------------

def find_next_pass(
    norad_id,
    search_hours=72
):

    satellites = get_satellites()

    if norad_id not in satellites:
        raise KeyError(f"Unknown object: {norad_id}")

    satellite = satellites[norad_id]["satellite"]

    now = datetime.now(timezone.utc)

    # If currently visible, begin immediately.
    current = get_reference_geometry(
        norad_id, now
    )

    if current["visible"]:
        return now

    t0 = ts.from_datetime(now)

    t1 = ts.from_datetime(
        now + timedelta(hours=search_hours)
    )

    times, events = satellite.find_events(
        GROUND_STATION,
        t0,
        t1,
        altitude_degrees=(
            STATION["minimum_elevation_deg"]
        )
    )

    for event_time, event in zip(times, events):

        # Event 0 = satellite rises.
        if event == 0:
            return (
                event_time.utc_datetime()
                + timedelta(seconds=5)
            )

    raise ValueError(
        "No visible pass found in search window"
    )


# ---------------------------------------
# Generate a sequence of observations
# ---------------------------------------

def simulate_sensor_series(
    norad_id,
    duration_seconds=300,
    step_seconds=10,
    start_time=None,
    noise_multiplier=1.0,
    dropout_sensor=None,
    seed=42
):

    if start_time is None:
        start_time = find_next_pass(norad_id)

    rng = np.random.default_rng(seed)

    measurements = []

    for seconds in range(
        0,
        duration_seconds + 1,
        step_seconds
    ):

        at_time = start_time + timedelta(
            seconds=seconds
        )

        reading = simulate_measurements(
            norad_id=norad_id,
            at_time=at_time,
            rng=rng,
            noise_multiplier=noise_multiplier,
            dropout_sensor=dropout_sensor
        )

        # Reference state for evaluating
        # fusion performance in Phase 4.
        state = get_state(norad_id, at_time)

        reading["reference_state"] = {
            "position_km": state["position_km"],
            "velocity_km_s": state["velocity_km_s"],
            "frame": "GCRS"
        }

        measurements.append(reading)

    return {
        "id": norad_id,
        "name": get_satellites()[norad_id][
            "satellite"
        ].name,
        "station": STATION,
        "start_time": start_time.isoformat(),
        "duration_seconds": duration_seconds,
        "step_seconds": step_seconds,
        "noise_multiplier": noise_multiplier,
        "dropout_sensor": dropout_sensor,
        "total_samples": len(measurements),
        "visible_samples": sum(
            m["visible"] for m in measurements
        ),
        "simulated": True,
        "reference_source": "CelesTrak + SGP4",
        "measurements": measurements
    }


# ---------------------------------------
# Test module
# ---------------------------------------

if __name__ == "__main__":

    import json

    satellites = get_satellites()

    norad_id = (
        25544 if 25544 in satellites
        else next(iter(satellites))
    )

    print("COSMOTRACK SENSOR SIMULATION")
    print("=" * 50)

    result = simulate_sensor_series(norad_id)

    print("Satellite:", result["name"])
    print("Station:", result["station"]["name"])
    print("Pass starts:", result["start_time"])
    print("Total samples:", result["total_samples"])
    print("Visible samples:", result["visible_samples"])

    print("\nFIRST MEASUREMENT:")
    print(json.dumps(
        result["measurements"][0],
        indent=2
    ))

    print("\nSensor simulation successful!")
