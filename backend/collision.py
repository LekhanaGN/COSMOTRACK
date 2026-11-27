
from datetime import datetime, timezone, timedelta
from itertools import combinations
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize_scalar

from backend.orbit import (
    get_satellites,
    get_state,
    ts
)


# ==========================================
# CONFIGURATION
# ==========================================

RISK_THRESHOLDS = {
    "critical_km": 1.0,
    "high_km": 5.0,
    "medium_km": 10.0
}

RISK_PRIORITY = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1
}


# ==========================================
# RISK CLASSIFICATION
# ==========================================

def classify_risk(miss_distance_km):

    if (
        not np.isfinite(miss_distance_km)
        or miss_distance_km < 0
    ):
        raise ValueError("Invalid miss distance")

    if miss_distance_km < 1.0:
        return "CRITICAL"

    elif miss_distance_km < 5.0:
        return "HIGH"

    elif miss_distance_km < 10.0:
        return "MEDIUM"

    return "LOW"


def recommended_action(risk):

    actions = {
        "CRITICAL": (
            "Prioritise immediate observation "
            "and independent orbit verification."
        ),

        "HIGH": (
            "Schedule additional observations "
            "and reassess the encounter."
        ),

        "MEDIUM": (
            "Increase monitoring frequency."
        ),

        "LOW": (
            "Continue routine surveillance."
        )
    }

    return actions[risk]


def generate_explanation(
    distance,
    relative_velocity,
    hours_to_tca,
    risk
):

    return (
        f"Predicted minimum separation is "
        f"{distance:.3f} km, with a relative "
        f"velocity of {relative_velocity:.3f} km/s. "
        f"Closest approach occurs in "
        f"{hours_to_tca:.2f} hours from the "
        f"screening start. "
        f"Assigned {risk} screening priority "
        f"using prototype distance thresholds."
    )


# ==========================================
# CLOSEST APPROACH OPTIMISATION
# ==========================================

def refine_closest_approach(
    position_function_a,
    position_function_b,
    lower_seconds,
    upper_seconds
):

    def squared_distance(seconds):

        relative_position = (
            position_function_a(seconds)
            - position_function_b(seconds)
        )

        return float(
            relative_position
            @ relative_position
        )

    if upper_seconds <= lower_seconds:

        best_time = float(lower_seconds)

    else:

        result = minimize_scalar(
            squared_distance,
            bounds=(
                lower_seconds,
                upper_seconds
            ),
            method="bounded",
            options={"xatol": 0.01}
        )

        candidates = [
            float(lower_seconds),
            float(upper_seconds)
        ]

        if result.success:
            candidates.append(float(result.x))

        best_time = min(
            candidates,
            key=squared_distance
        )

    miss_distance = np.sqrt(
        squared_distance(best_time)
    )

    return best_time, float(miss_distance)


# ==========================================
# ALERT STRUCTURE
# ==========================================

def build_alert(
    object_a,
    object_b,
    start_time,
    tca_seconds,
    miss_distance,
    relative_velocity,
    position_a,
    position_b,
    simulated=False
):

    risk = classify_risk(miss_distance)

    tca = start_time + timedelta(
        seconds=float(tca_seconds)
    )

    hours_to_tca = tca_seconds / 3600

    return {
        "alert_id": (
            f"{object_a['id']}-"
            f"{object_b['id']}-"
            f"{int(tca.timestamp())}"
        ),

        "object_a": object_a,
        "object_b": object_b,

        "tca": tca.isoformat(),

        "seconds_to_tca":
            float(tca_seconds),

        "hours_to_tca":
            float(hours_to_tca),

        "miss_distance_km":
            float(miss_distance),

        "relative_velocity_km_s":
            float(relative_velocity),

        "risk_level": risk,

        "priority":
            RISK_PRIORITY[risk],

        "position_a_km":
            np.asarray(position_a).tolist(),

        "position_b_km":
            np.asarray(position_b).tolist(),

        "position_frame": "GCRS",

        "explanation": generate_explanation(
            miss_distance,
            relative_velocity,
            hours_to_tca,
            risk
        ),

        "recommended_action":
            recommended_action(risk),

        "collision_probability": None,

        "simulated": simulated,

        "assessment_method":
            "Distance-based conjunction screening"
    }


# ==========================================
# OBJECT METADATA
# ==========================================

def object_metadata(norad_id):

    item = get_satellites()[norad_id]

    return {
        "id": norad_id,

        "name":
            item["metadata"]["OBJECT_NAME"],

        "type":
            item["metadata"]["COSMOTRACK_TYPE"],

        "catalog_epoch":
            item["metadata"]["EPOCH"]
    }


# ==========================================
# REAL CATALOGUE SCREENING
# ==========================================

def screen_conjunctions(
    hours=24,
    step_seconds=60,
    screening_distance_km=25.0,
    max_alerts=50,
    start_time=None
):

    if hours <= 0:
        raise ValueError("Hours must be positive")

    if step_seconds <= 0:
        raise ValueError("Step must be positive")

    if screening_distance_km <= 0:
        raise ValueError(
            "Screening distance must be positive"
        )

    if start_time is None:
        start_time = datetime.now(timezone.utc)

    satellites = get_satellites()

    duration_seconds = int(hours * 3600)

    offsets = np.arange(
        0,
        duration_seconds,
        step_seconds,
        dtype=float
    )

    offsets = np.append(
        offsets,
        duration_seconds
    )

    timestamps = [
        start_time + timedelta(seconds=float(s))
        for s in offsets
    ]

    times = ts.from_datetimes(timestamps)

    # Vectorised propagation of every object.
    positions = {}

    skipped_objects = []

    for norad_id, item in satellites.items():

        satellite = item["satellite"]

        states = satellite.at(times)

        position_array = (
            states.position.km.T
        )

        if not np.isfinite(position_array).all():

            skipped_objects.append(norad_id)

            continue

        positions[norad_id] = position_array

    ids = list(positions.keys())

    alerts = []

    screened_pairs = 0

    refined_candidates = 0

    # Conservative coarse screening margin
    # based on an assumed maximum relative
    # speed of 20 km/s for LEO encounters.
    coarse_margin = (
        20.0 * step_seconds / 2
    )

    coarse_threshold = (
        screening_distance_km
        + coarse_margin
    )

    for id_a, id_b in combinations(ids, 2):

        metadata_a = object_metadata(id_a)

        metadata_b = object_metadata(id_b)

        # Ignore debris-to-debris encounters.
        if (
            metadata_a["type"] == "DEBRIS"
            and metadata_b["type"] == "DEBRIS"
        ):
            continue

        screened_pairs += 1

        relative_positions = (
            positions[id_a]
            - positions[id_b]
        )

        distances = np.linalg.norm(
            relative_positions,
            axis=1
        )

        # Find sampled local minima.
        candidate_indices = []

        if distances[0] <= distances[1]:
            candidate_indices.append(0)

        for i in range(
            1, len(distances) - 1
        ):

            if (
                distances[i] <= distances[i - 1]
                and
                distances[i] <= distances[i + 1]
            ):

                candidate_indices.append(i)

        if distances[-1] <= distances[-2]:

            candidate_indices.append(
                len(distances) - 1
            )

        satellite_a = satellites[id_a][
            "satellite"
        ]

        satellite_b = satellites[id_b][
            "satellite"
        ]

        def position_a(seconds):

            t = ts.from_datetime(
                start_time + timedelta(
                    seconds=float(seconds)
                )
            )

            return satellite_a.at(t).position.km

        def position_b(seconds):

            t = ts.from_datetime(
                start_time + timedelta(
                    seconds=float(seconds)
                )
            )

            return satellite_b.at(t).position.km

        pair_alerts = []

        for index in candidate_indices:

            if distances[index] > coarse_threshold:
                continue

            lower = offsets[
                max(0, index - 1)
            ]

            upper = offsets[
                min(len(offsets) - 1, index + 1)
            ]

            refined_candidates += 1

            tca_seconds, miss_distance = (
                refine_closest_approach(
                    position_a,
                    position_b,
                    lower,
                    upper
                )
            )

            if not np.isfinite(miss_distance):
                continue

            if miss_distance > screening_distance_km:
                continue

            tca = start_time + timedelta(
                seconds=tca_seconds
            )

            t = ts.from_datetime(tca)

            state_a = satellite_a.at(t)

            state_b = satellite_b.at(t)

            relative_velocity = np.linalg.norm(
                state_a.velocity.km_per_s
                - state_b.velocity.km_per_s
            )

            if not np.isfinite(relative_velocity):
                continue

            alert = build_alert(
                object_a=metadata_a,
                object_b=metadata_b,
                start_time=start_time,
                tca_seconds=tca_seconds,
                miss_distance=miss_distance,
                relative_velocity=relative_velocity,
                position_a=state_a.position.km,
                position_b=state_b.position.km,
                simulated=False
            )

            # Flag minima at screening boundaries.
            alert["window_boundary"] = bool(
                tca_seconds < 0.1
                or
                abs(
                    tca_seconds - duration_seconds
                ) < 0.1
            )

            # Avoid duplicate alerts for
            # the same encounter.
            duplicate = any(
                abs(
                    existing["seconds_to_tca"]
                    - tca_seconds
                ) < step_seconds
                for existing in pair_alerts
            )

            if not duplicate:
                pair_alerts.append(alert)

        alerts.extend(pair_alerts)

    # Highest priority first, then urgency.
    alerts.sort(
        key=lambda alert: (
            -alert["priority"],
            alert["seconds_to_tca"],
            alert["miss_distance_km"]
        )
    )

    risk_counts = {
        "CRITICAL": 0,
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0
    }

    for alert in alerts:

        risk_counts[
            alert["risk_level"]
        ] += 1

    return {
        "mode": "catalogue_prediction",

        "source": "CelesTrak + SGP4",

        "screening_start":
            start_time.isoformat(),

        "screening_end": (
            start_time
            + timedelta(hours=hours)
        ).isoformat(),

        "prediction_hours": hours,

        "step_seconds": step_seconds,

        "screening_distance_km":
            screening_distance_km,

        "total_objects": len(satellites),

        "valid_objects": len(ids),

        "skipped_objects": skipped_objects,

        "screened_pairs": screened_pairs,

        "refined_candidates":
            refined_candidates,

        "total_alerts": len(alerts),

        "returned_alerts":
            min(len(alerts), max_alerts),

        "risk_counts": risk_counts,

        "risk_thresholds": RISK_THRESHOLDS,

        "collision_probability_available":
            False,

        "alerts": alerts[:max_alerts]
    }


# ==========================================
# CACHED SCREENING FOR DASHBOARD
# ==========================================

@lru_cache(maxsize=8)
def cached_screening(
    hours,
    step_seconds,
    screening_distance_km,
    time_bucket
):

    start_time = datetime.fromtimestamp(
        time_bucket * 300,
        tz=timezone.utc
    )

    return screen_conjunctions(
        hours=hours,
        step_seconds=step_seconds,
        screening_distance_km=(
            screening_distance_km
        ),
        start_time=start_time
    )


def get_conjunctions(
    hours=24,
    step_seconds=60,
    screening_distance_km=25.0
):

    # Refresh screening every five minutes.
    time_bucket = int(
        datetime.now(timezone.utc).timestamp()
        // 300
    )

    return cached_screening(
        hours,
        step_seconds,
        screening_distance_km,
        time_bucket
    )


# ==========================================
# SYNTHETIC CLOSE-APPROACH SCENARIO
# ==========================================

def synthetic_encounter(
    miss_distance_km=0.45,
    relative_speed_km_s=12.0,
    tca_seconds=600,
    norad_id=25544,
    start_time=None
):

    if miss_distance_km < 0:
        raise ValueError(
            "Miss distance cannot be negative"
        )

    if relative_speed_km_s <= 0:
        raise ValueError(
            "Relative speed must be positive"
        )

    if tca_seconds <= 0:
        raise ValueError(
            "TCA must be in the future"
        )

    satellites = get_satellites()

    if norad_id not in satellites:

        raise KeyError(
            f"Unknown satellite: {norad_id}"
        )

    if start_time is None:
        start_time = datetime.now(timezone.utc)

    encounter_time = (
        start_time
        + timedelta(seconds=tca_seconds)
    )

    state = get_state(
        norad_id,
        encounter_time
    )

    r = np.array(state["position_km"])

    v = np.array(state["velocity_km_s"])

    # Construct an orthogonal local frame.
    radial = r / np.linalg.norm(r)

    normal = np.cross(r, v)

    normal = normal / np.linalg.norm(normal)

    # Local linear relative motion model.
    # Valid only as a short encounter demo.
    def position_a(seconds):

        dt = seconds - tca_seconds

        return r + v * dt

    def position_b(seconds):

        dt = seconds - tca_seconds

        return (
            position_a(seconds)
            + radial * miss_distance_km
            + normal * relative_speed_km_s * dt
        )

    refined_time, refined_distance = (
        refine_closest_approach(
            position_a,
            position_b,
            max(0, tca_seconds - 30),
            tca_seconds + 30
        )
    )

    metadata_a = object_metadata(norad_id)

    metadata_b = {
        "id": "SYN-DEBRIS-001",
        "name": "Synthetic Debris Fragment",
        "type": "DEBRIS"
    }

    alert = build_alert(
        object_a=metadata_a,
        object_b=metadata_b,
        start_time=start_time,
        tca_seconds=refined_time,
        miss_distance=refined_distance,
        relative_velocity=relative_speed_km_s,
        position_a=position_a(refined_time),
        position_b=position_b(refined_time),
        simulated=True
    )

    trajectory = []

    # Dense sampling around the encounter.
    # Include the exact TCA.
    offsets = np.unique(
        np.concatenate([
            np.linspace(-30, 30, 241),
            np.array([0.0])
        ])
    )

    for offset in offsets:

        seconds = refined_time + float(offset)

        pa = position_a(seconds)

        pb = position_b(seconds)

        trajectory.append({
            "timestamp": (
                start_time
                + timedelta(seconds=seconds)
            ).isoformat(),

            "seconds_from_tca":
                float(offset),

            "position_a_km": pa.tolist(),

            "position_b_km": pb.tolist(),

            "relative_position_km":
                (pb - pa).tolist(),

            "separation_km": float(
                np.linalg.norm(pb - pa)
            )
        })

    return {
        "mode": "synthetic_encounter",

        "simulated": True,

        "description": (
            "Controlled local linear encounter "
            "anchored to an SGP4 satellite state. "
            "The debris object is fictional."
        ),

        "alert": alert,

        "trajectory": trajectory
    }


# ==========================================
# MODULE TEST
# ==========================================

if __name__ == "__main__":

    import json
    import time

    print("COSMOTRACK CONJUNCTION ENGINE")
    print("=" * 55)

    print("\nSYNTHETIC ENCOUNTER:")

    demo = synthetic_encounter()

    print(json.dumps(
        demo["alert"],
        indent=2
    ))

    print("\nREAL CATALOGUE SCREENING:")

    started = time.perf_counter()

    result = screen_conjunctions(
        hours=6,
        step_seconds=60,
        screening_distance_km=25
    )

    print(
        "Objects:",
        result["valid_objects"]
    )

    print(
        "Pairs screened:",
        result["screened_pairs"]
    )

    print(
        "Total alerts:",
        result["total_alerts"]
    )

    print(
        "Risk counts:",
        result["risk_counts"]
    )

    for alert in result["alerts"][:5]:

        print(
            alert["object_a"]["name"],
            "<->",
            alert["object_b"]["name"],
            "|",
            alert["risk_level"],
            "|",
            round(
                alert["miss_distance_km"], 3
            ),
            "km"
        )

    print(
        "\nExecution time:",
        round(time.perf_counter() - started, 2),
        "seconds"
    )

    print("\nConjunction engine completed!")
