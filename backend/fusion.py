
from datetime import datetime

import numpy as np

from scipy.integrate import solve_ivp
from skyfield.framelib import itrs

from backend.orbit import ts, get_satellites

from backend.sensors import (
    simulate_sensor_series,
    STATION
)


# ==========================================
# CONSTANTS
# ==========================================

MU = 398600.4418  # Earth's gravity, km^3/s^2

STATE_SIZE = 6

DEG_TO_RAD = np.pi / 180.0

# Assumed unmodelled acceleration noise.
ACCEL_NOISE = 1e-5  # km/s^2


# ==========================================
# ORBITAL DYNAMICS
# ==========================================

def orbital_dynamics(t, state):

    position = state[:3]
    velocity = state[3:]

    radius = np.linalg.norm(position)

    acceleration = (
        -MU * position / radius**3
    )

    return np.concatenate([
        velocity,
        acceleration
    ])


def propagate_state(state, dt):

    if dt == 0:
        return state.copy()

    solution = solve_ivp(
        orbital_dynamics,
        (0, dt),
        state,
        method="RK45",
        rtol=1e-9,
        atol=1e-10
    )

    if not solution.success:
        raise RuntimeError(
            "Orbital propagation failed"
        )

    return solution.y[:, -1]


# ==========================================
# NUMERICAL JACOBIAN
# ==========================================

def numerical_jacobian(function, state):

    state = np.asarray(state, dtype=float)

    output = np.asarray(function(state))

    jacobian = np.zeros(
        (output.size, state.size)
    )

    for i in range(state.size):

        epsilon = (
            1e-3 if i < 3 else 1e-6
        )

        plus = state.copy()
        minus = state.copy()

        plus[i] += epsilon
        minus[i] -= epsilon

        difference = (
            function(plus)
            - function(minus)
        )

        jacobian[:, i] = (
            difference / (2 * epsilon)
        )

    return jacobian


# ==========================================
# PROCESS NOISE COVARIANCE
# ==========================================

def process_noise(dt):

    identity = np.eye(3)

    q = ACCEL_NOISE**2

    return q * np.block([
        [
            (dt**4 / 4) * identity,
            (dt**3 / 2) * identity
        ],
        [
            (dt**3 / 2) * identity,
            dt**2 * identity
        ]
    ])


# ==========================================
# GROUND STATION GEOMETRY
# ==========================================

def station_geometry(reference):

    at_time = datetime.fromisoformat(
        reference["timestamp"]
    )

    t = ts.from_datetime(at_time)

    # GCRS -> Earth-fixed rotation.
    rotation = itrs.rotation_at(t)

    latitude = (
        STATION["latitude_deg"] * DEG_TO_RAD
    )

    longitude = (
        STATION["longitude_deg"] * DEG_TO_RAD
    )

    # Earth-fixed -> local East-North-Up.
    enu_rotation = np.array([
        [
            -np.sin(longitude),
            np.cos(longitude),
            0
        ],
        [
            -np.sin(latitude)
            * np.cos(longitude),

            -np.sin(latitude)
            * np.sin(longitude),

            np.cos(latitude)
        ],
        [
            np.cos(latitude)
            * np.cos(longitude),

            np.cos(latitude)
            * np.sin(longitude),

            np.sin(latitude)
        ]
    ])

    return {
        "position": np.array(
            reference["station_position_km"]
        ),
        "velocity": np.array(
            reference["station_velocity_km_s"]
        ),
        "gcrs_to_enu": enu_rotation @ rotation
    }


# ==========================================
# NONLINEAR MEASUREMENT MODEL
# ==========================================

def measurement_model(
    state,
    geometry,
    measurement_types
):

    relative_position = (
        state[:3] - geometry["position"]
    )

    relative_velocity = (
        state[3:] - geometry["velocity"]
    )

    range_km = np.linalg.norm(
        relative_position
    )

    range_rate = np.dot(
        relative_position,
        relative_velocity
    ) / range_km

    east, north, up = (
        geometry["gcrs_to_enu"]
        @ relative_position
    )

    azimuth = np.arctan2(east, north)

    elevation = np.arctan2(
        up,
        np.hypot(east, north)
    )

    values = {
        "lidar_range": range_km,
        "radar_range": range_km,
        "radar_range_rate": range_rate,
        "optical_azimuth": azimuth,
        "optical_elevation": elevation
    }

    return np.array([
        values[name]
        for name in measurement_types
    ])


# ==========================================
# BUILD MEASUREMENT VECTOR AND R MATRIX
# ==========================================

def build_measurement(reading):

    values = []
    variances = []
    types = []

    lidar = reading["lidar"]
    radar = reading["radar"]
    optical = reading["optical"]

    if lidar is not None:

        values.append(lidar["range_km"])

        variances.append(
            lidar["sigma_range_km"]**2
        )

        types.append("lidar_range")

    if radar is not None:

        values.extend([
            radar["range_km"],
            radar["range_rate_km_s"]
        ])

        variances.extend([
            radar["sigma_range_km"]**2,
            radar["sigma_range_rate_km_s"]**2
        ])

        types.extend([
            "radar_range",
            "radar_range_rate"
        ])

    if optical is not None:

        values.extend([
            optical["azimuth_deg"]
            * DEG_TO_RAD,

            optical["elevation_deg"]
            * DEG_TO_RAD
        ])

        variances.extend([
            (
                optical["sigma_azimuth_deg"]
                * DEG_TO_RAD
            )**2,

            (
                optical["sigma_elevation_deg"]
                * DEG_TO_RAD
            )**2
        ])

        types.extend([
            "optical_azimuth",
            "optical_elevation"
        ])

    return (
        np.array(values),
        np.diag(variances),
        types
    )


def wrap_angle(angle):

    return (
        (angle + np.pi)
        % (2 * np.pi)
    ) - np.pi


# ==========================================
# MEASUREMENT JACOBIAN
# ==========================================

def measurement_jacobian(
    state,
    geometry,
    types
):

    jacobian = np.zeros(
        (len(types), STATE_SIZE)
    )

    for i in range(STATE_SIZE):

        epsilon = (
            1e-3 if i < 3 else 1e-6
        )

        plus = state.copy()
        minus = state.copy()

        plus[i] += epsilon
        minus[i] -= epsilon

        difference = (
            measurement_model(
                plus, geometry, types
            )
            -
            measurement_model(
                minus, geometry, types
            )
        )

        # Prevent azimuth discontinuities
        # near the 0/360-degree boundary.
        for j, name in enumerate(types):

            if name == "optical_azimuth":
                difference[j] = wrap_angle(
                    difference[j]
                )

        jacobian[:, i] = (
            difference / (2 * epsilon)
        )

    return jacobian


# ==========================================
# EXTENDED KALMAN FILTER
# ==========================================

class OrbitalEKF:

    def __init__(self, initial_state):

        self.x = np.array(
            initial_state, dtype=float
        )

        # Initial position uncertainty: 2 km
        # Initial velocity uncertainty: 20 m/s
        self.P = np.diag([
            2.0**2,
            2.0**2,
            2.0**2,
            0.02**2,
            0.02**2,
            0.02**2
        ])

    def predict(self, dt):

        if dt <= 0:
            return

        previous_state = self.x.copy()

        transition = lambda state: (
            propagate_state(state, dt)
        )

        F = numerical_jacobian(
            transition,
            previous_state
        )

        self.x = transition(previous_state)

        Q = process_noise(dt)

        self.P = (
            F @ self.P @ F.T + Q
        )

        self.P = (
            self.P + self.P.T
        ) / 2

    def update(self, reading):

        z, R, types = build_measurement(
            reading
        )

        # No measurements available.
        if len(types) == 0:
            return {
                "updated": False,
                "measurement_count": 0,
                "nis": None
            }

        geometry = station_geometry(
            reading["reference"]
        )

        predicted_measurement = (
            measurement_model(
                self.x,
                geometry,
                types
            )
        )

        H = measurement_jacobian(
            self.x,
            geometry,
            types
        )

        innovation = (
            z - predicted_measurement
        )

        for i, name in enumerate(types):

            if name == "optical_azimuth":
                innovation[i] = wrap_angle(
                    innovation[i]
                )

        S = (
            H @ self.P @ H.T + R
        )

        # Numerically stable Kalman gain.
        K = np.linalg.solve(
            S,
            H @ self.P
        ).T

        # State correction.
        self.x = self.x + K @ innovation

        # Joseph covariance update.
        # Helps maintain numerical stability.
        I = np.eye(STATE_SIZE)

        A = I - K @ H

        self.P = (
            A @ self.P @ A.T
            + K @ R @ K.T
        )

        self.P = (
            self.P + self.P.T
        ) / 2

        # Normalised Innovation Squared.
        nis = float(
            innovation.T
            @ np.linalg.solve(
                S, innovation
            )
        )

        return {
            "updated": True,
            "measurement_count": len(types),
            "measurement_types": types,
            "nis": nis,
            "innovation": innovation.tolist()
        }


# ==========================================
# RMSE CALCULATION
# ==========================================

def calculate_rmse(errors):

    errors = np.asarray(errors)

    return float(
        np.sqrt(np.mean(errors**2))
    )


# ==========================================
# COMPLETE SENSOR FUSION EXPERIMENT
# ==========================================

def run_fusion(
    norad_id,
    duration_seconds=300,
    step_seconds=10,
    noise_multiplier=1.0,
    dropout_sensor=None,
    seed=42,
    start_time=None
):

    sensor_data = simulate_sensor_series(
        norad_id=norad_id,
        duration_seconds=duration_seconds,
        step_seconds=step_seconds,
        noise_multiplier=noise_multiplier,
        dropout_sensor=dropout_sensor,
        seed=seed,
        start_time=start_time
    )

    measurements = sensor_data["measurements"]

    if not measurements:
        raise ValueError(
            "No sensor samples available"
        )

    first_reference = measurements[0][
        "reference_state"
    ]

    initial_reference = np.array(
        first_reference["position_km"]
        + first_reference["velocity_km_s"]
    )

    # Controlled initial tracking error.
    rng = np.random.default_rng(seed + 1)

    initial_error = np.concatenate([
        rng.normal(0, 2.0, 3),
        rng.normal(0, 0.02, 3)
    ])

    initial_state = (
        initial_reference + initial_error
    )

    ekf = OrbitalEKF(initial_state)

    # Baseline without sensor corrections.
    baseline_state = initial_state.copy()

    previous_time = datetime.fromisoformat(
        measurements[0]["timestamp"]
    )

    results = []

    baseline_errors = []
    fused_errors = []

    baseline_velocity_errors = []
    fused_velocity_errors = []

    for reading in measurements:

        current_time = datetime.fromisoformat(
            reading["timestamp"]
        )

        dt = (
            current_time - previous_time
        ).total_seconds()

        reference = reading["reference_state"]

        truth = np.array(
            reference["position_km"]
            + reference["velocity_km_s"]
        )

        # Predict both trajectories.
        if dt > 0:

            baseline_state = propagate_state(
                baseline_state, dt
            )

            ekf.predict(dt)

        # Correct only the EKF trajectory.
        update_info = ekf.update(reading)

        baseline_error = float(
            np.linalg.norm(
                baseline_state[:3]
                - truth[:3]
            )
        )

        fused_error = float(
            np.linalg.norm(
                ekf.x[:3] - truth[:3]
            )
        )

        baseline_velocity_error = float(
            np.linalg.norm(
                baseline_state[3:]
                - truth[3:]
            )
        )

        fused_velocity_error = float(
            np.linalg.norm(
                ekf.x[3:] - truth[3:]
            )
        )

        baseline_errors.append(
            baseline_error
        )

        fused_errors.append(fused_error)

        baseline_velocity_errors.append(
            baseline_velocity_error
        )

        fused_velocity_errors.append(
            fused_velocity_error
        )

        position_covariance = ekf.P[:3, :3]

        position_sigma = float(
            np.sqrt(
                max(
                    np.trace(
                        position_covariance
                    ),
                    0
                )
            )
        )

        velocity_sigma = float(
            np.sqrt(
                max(
                    np.trace(ekf.P[3:, 3:]),
                    0
                )
            )
        )

        results.append({
            "timestamp": reading["timestamp"],
            "elapsed_seconds": (
                current_time
                - datetime.fromisoformat(
                    measurements[0]["timestamp"]
                )
            ).total_seconds(),

            "reference_position_km":
                truth[:3].tolist(),

            "reference_velocity_km_s":
                truth[3:].tolist(),

            "baseline_position_km":
                baseline_state[:3].tolist(),

            "fused_position_km":
                ekf.x[:3].tolist(),

            "fused_velocity_km_s":
                ekf.x[3:].tolist(),

            "baseline_error_km":
                baseline_error,

            "fused_error_km":
                fused_error,

            "baseline_velocity_error_km_s":
                baseline_velocity_error,

            "fused_velocity_error_km_s":
                fused_velocity_error,

            "position_uncertainty_km":
                position_sigma,

            "velocity_uncertainty_km_s":
                velocity_sigma,

            "covariance": ekf.P.tolist(),

            "sensor_update":
                update_info["updated"],

            "measurement_count":
                update_info["measurement_count"],

            "nis": update_info["nis"],

            "visible": reading["visible"]
        })

        previous_time = current_time

    baseline_rmse = calculate_rmse(
        baseline_errors
    )

    fused_rmse = calculate_rmse(
        fused_errors
    )

    improvement = (
        100 * (baseline_rmse - fused_rmse)
        / baseline_rmse
        if baseline_rmse > 0
        else 0.0
    )

    return {
        "id": norad_id,
        "name": sensor_data["name"],
        "algorithm": "Extended Kalman Filter",
        "dynamics": "Two-body orbital model",
        "frame": "GCRS",
        "simulated": True,
        "reference_source": "CelesTrak + SGP4",
        "station": sensor_data["station"],
        "start_time": sensor_data["start_time"],
        "noise_multiplier": noise_multiplier,
        "dropout_sensor": dropout_sensor,
        "total_samples": len(results),
        "measurement_updates": sum(
            r["sensor_update"]
            for r in results
        ),

        "metrics": {
            "baseline_rmse_km":
                baseline_rmse,

            "fused_rmse_km":
                fused_rmse,

            "improvement_percent":
                improvement,

            "baseline_velocity_rmse_km_s":
                calculate_rmse(
                    baseline_velocity_errors
                ),

            "fused_velocity_rmse_km_s":
                calculate_rmse(
                    fused_velocity_errors
                ),

            "initial_position_error_km":
                float(
                    np.linalg.norm(
                        initial_error[:3]
                    )
                ),

            "final_position_error_km":
                results[-1]["fused_error_km"],

            "final_uncertainty_km":
                results[-1][
                    "position_uncertainty_km"
                ]
        },

        "final_state": {
            "timestamp": results[-1]["timestamp"],
            "position_km": ekf.x[:3].tolist(),
            "velocity_km_s": ekf.x[3:].tolist(),
            "covariance": ekf.P.tolist()
        },

        "results": results
    }


# ==========================================
# MODULE TEST
# ==========================================

if __name__ == "__main__":

    import json

    satellites = get_satellites()

    norad_id = (
        25544 if 25544 in satellites
        else next(iter(satellites))
    )

    print("COSMOTRACK SENSOR FUSION")
    print("=" * 50)

    result = run_fusion(norad_id)

    print("Satellite:", result["name"])

    print(
        "Measurement updates:",
        result["measurement_updates"]
    )

    print("\nPERFORMANCE METRICS:")

    print(json.dumps(
        result["metrics"],
        indent=2
    ))

    print("\nFINAL STATE:")

    print(
        "Position:",
        result["final_state"]["position_km"]
    )

    print(
        "Velocity:",
        result["final_state"]["velocity_km_s"]
    )

    print("\nEKF execution completed!")
