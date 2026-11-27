
from datetime import datetime, timezone, timedelta

import numpy as np

from backend.orbit import (
    get_satellites,
    ts
)

from backend.sensors import (
    GROUND_STATION
)

from backend.collision import (
    get_conjunctions,
    synthetic_encounter
)

from backend.anomaly import (
    run_anomaly_demo
)

from backend.fusion import (
    run_fusion
)


# ==========================================
# SCHEDULER CONFIGURATION
# ==========================================

PRIORITY_WEIGHTS = {
    "conjunction": 35.0,
    "anomaly": 25.0,
    "uncertainty": 20.0,
    "visibility": 10.0,
    "urgency": 10.0
}

SEVERITY_SCORES = {
    "NORMAL": 0.0,
    "LOW": 0.25,
    "MEDIUM": 0.50,
    "HIGH": 0.75,
    "CRITICAL": 1.0
}

# Configurable simulated sensor capabilities.
# These are not validated hardware limits.

SENSOR_CONFIG = {
    "radar": {
        "name": "Radar",
        "min_elevation_deg": 10.0,
        "max_range_km": 2500.0,
        "duration_seconds": 60
    },

    "lidar": {
        "name": "LiDAR",
        "min_elevation_deg": 20.0,
        "max_range_km": 1500.0,
        "duration_seconds": 30
    },

    "optical": {
        "name": "Optical",
        "min_elevation_deg": 15.0,
        "max_range_km": 2500.0,
        "duration_seconds": 60
    }
}

UNCERTAINTY_REFERENCE_KM = 2.0

SENSOR_SETUP_SECONDS = 10

VISIBILITY_STEP_SECONDS = 10

DEFAULT_NORAD_ID = 25544


# ==========================================
# GENERAL UTILITIES
# ==========================================

def parse_timestamp(value):

    if isinstance(value, datetime):
        result = value

    else:
        result = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

    if result.tzinfo is None:
        raise ValueError(
            "Timestamp must be timezone-aware"
        )

    return result.astimezone(timezone.utc)


def clip_score(value):

    value = float(value)

    if not np.isfinite(value):
        raise ValueError(
            "Score must be finite"
        )

    return float(
        np.clip(value, 0.0, 1.0)
    )


def severity_score(severity):

    return SEVERITY_SCORES.get(
        str(severity).upper(),
        0.0
    )


# ==========================================
# PRIORITY CALCULATION
# ==========================================

def calculate_priority(
    conjunction=0.0,
    anomaly=0.0,
    uncertainty=0.0,
    visibility=0.0,
    urgency=0.0
):

    factors = {
        "conjunction":
            clip_score(conjunction),

        "anomaly":
            clip_score(anomaly),

        "uncertainty":
            clip_score(uncertainty),

        "visibility":
            clip_score(visibility),

        "urgency":
            clip_score(urgency)
    }

    contributions = {
        name: round(
            factors[name] * weight,
            4
        )

        for name, weight
        in PRIORITY_WEIGHTS.items()
    }

    priority = round(
        sum(contributions.values()),
        2
    )

    return {
        "priority_score": priority,
        "priority_level":
            classify_priority(priority),
        "factors": factors,
        "contributions": contributions
    }


def classify_priority(score):

    if score >= 80:
        return "CRITICAL"

    if score >= 60:
        return "HIGH"

    if score >= 40:
        return "MEDIUM"

    return "LOW"


def calculate_urgency(
    tca,
    start_time,
    horizon_hours=24
):

    if tca is None:
        return 0.0

    tca = parse_timestamp(tca)

    start_time = parse_timestamp(
        start_time
    )

    seconds = (
        tca - start_time
    ).total_seconds()

    if seconds < 0:
        return 0.0

    return clip_score(
        1.0
        - seconds
        / (horizon_hours * 3600)
    )


def calculate_uncertainty_score(
    uncertainty_km
):

    if uncertainty_km is None:
        return 0.0

    if uncertainty_km < 0:
        raise ValueError(
            "Uncertainty cannot be negative"
        )

    return clip_score(
        uncertainty_km
        / UNCERTAINTY_REFERENCE_KM
    )


# ==========================================
# SENSOR PREFERENCE
# ==========================================

def preferred_sensors(
    anomaly_type="NORMAL",
    conjunction_risk="LOW",
    uncertainty_km=None
):

    # Prefer independent sensors when
    # investigating a suspected sensor fault.

    preferences = {
        "LIDAR_ANOMALY": [
            "radar",
            "optical"
        ],

        "RADAR_ANOMALY": [
            "lidar",
            "optical"
        ],

        "OPTICAL_ANOMALY": [
            "radar",
            "lidar"
        ],

        "VELOCITY_ANOMALY": [
            "radar",
            "lidar",
            "optical"
        ],

        "ORBITAL_DEVIATION": [
            "radar",
            "lidar",
            "optical"
        ]
    }

    if anomaly_type in preferences:

        return preferences[
            anomaly_type
        ].copy()

    if conjunction_risk in [
        "CRITICAL",
        "HIGH"
    ]:

        return [
            "radar",
            "lidar",
            "optical"
        ]

    if (
        uncertainty_km is not None
        and uncertainty_km >= 1.0
    ):

        return [
            "radar",
            "lidar",
            "optical"
        ]

    return [
        "optical",
        "radar",
        "lidar"
    ]


# ==========================================
# SATELLITE VISIBILITY PREDICTION
# ==========================================

def predict_visibility(
    norad_id,
    start_time,
    hours=6,
    step_seconds=VISIBILITY_STEP_SECONDS
):

    if hours <= 0 or step_seconds <= 0:
        raise ValueError(
            "Hours and step must be positive"
        )

    satellites = get_satellites()

    if norad_id not in satellites:
        raise KeyError(
            f"Unknown satellite: {norad_id}"
        )

    start_time = parse_timestamp(
        start_time
    )

    satellite = satellites[
        norad_id
    ]["satellite"]

    duration_seconds = int(
        hours * 3600
    )

    offsets = np.arange(
        0,
        duration_seconds + 1,
        step_seconds
    )

    if offsets[-1] != duration_seconds:

        offsets = np.append(
            offsets,
            duration_seconds
        )

    datetimes = [
        start_time
        + timedelta(seconds=int(offset))

        for offset in offsets
    ]

    times = ts.from_datetimes(
        datetimes
    )

    # Vectorised topocentric propagation.
    # Much faster than calling get_state()
    # individually for every timestamp.

    topocentric = (
        satellite - GROUND_STATION
    ).at(times)

    altitude, azimuth, distance = (
        topocentric.altaz()
    )

    elevations = np.asarray(
        altitude.degrees
    )

    ranges = np.asarray(
        distance.km
    )

    windows = []

    # Build separate visibility windows
    # for each sensor.

    for sensor, config in SENSOR_CONFIG.items():

        valid = (

            np.isfinite(elevations)

            & np.isfinite(ranges)

            & (
                elevations
                >= config[
                    "min_elevation_deg"
                ]
            )

            & (
                ranges
                <= config[
                    "max_range_km"
                ]
            )
        )

        indices = np.flatnonzero(
            valid
        )

        if len(indices) == 0:
            continue

        groups = np.split(
            indices,
            np.where(
                np.diff(indices) > 1
            )[0] + 1
        )

        for group in groups:

            first = int(group[0])

            last = int(group[-1])

            window_start = datetimes[
                first
            ]

            window_end = datetimes[
                last
            ]

            duration = (
                window_end - window_start
            ).total_seconds()

            if duration < config[
                "duration_seconds"
            ]:
                continue

            windows.append({

                "norad_id":
                    norad_id,

                "sensor":
                    sensor,

                "start_time":
                    window_start.isoformat(),

                "end_time":
                    window_end.isoformat(),

                "duration_seconds":
                    float(duration),

                "max_elevation_deg":
                    float(
                        np.max(
                            elevations[group]
                        )
                    ),

                "min_range_km":
                    float(
                        np.min(
                            ranges[group]
                        )
                    ),

                "visibility_method":
                    "Sampled SGP4 topocentric geometry"
            })

    windows.sort(
        key=lambda item:
            item["start_time"]
    )

    return windows


# ==========================================
# COLLECT CONJUNCTION INFORMATION
# ==========================================

def collect_conjunction_risks(
    alerts,
    start_time
):

    start_time = parse_timestamp(
        start_time
    )

    risks = {}

    for alert in alerts:

        risk = alert.get(
            "risk_level",
            "LOW"
        )

        tca = alert.get("tca")

        if tca is not None:

            if parse_timestamp(tca) < start_time:
                continue

        for object_key in [
            "object_a",
            "object_b"
        ]:

            obj = alert.get(
                object_key,
                {}
            )

            norad_id = obj.get(
                "norad_id"
            )

            # Ignore fictional objects.
            # Only catalogue targets can
            # be assigned observations.

            if not isinstance(
                norad_id,
                (int, np.integer)
            ):
                continue

            norad_id = int(norad_id)

            candidate = {
                "risk_level": risk,

                "miss_distance_km":
                    alert.get(
                        "miss_distance_km"
                    ),

                "tca": tca,

                "simulated":
                    bool(
                        alert.get(
                            "simulated",
                            False
                        )
                    )
            }

            previous = risks.get(
                norad_id
            )

            replace = previous is None

            if previous is not None:

                new_score = severity_score(
                    risk
                )

                old_score = severity_score(
                    previous["risk_level"]
                )

                if new_score > old_score:

                    replace = True

                elif new_score == old_score:

                    new_tca = (
                        parse_timestamp(tca)
                        if tca
                        else datetime.max.replace(
                            tzinfo=timezone.utc
                        )
                    )

                    old_tca = (
                        parse_timestamp(
                            previous["tca"]
                        )
                        if previous["tca"]
                        else datetime.max.replace(
                            tzinfo=timezone.utc
                        )
                    )

                    replace = new_tca < old_tca

            if replace:

                risks[norad_id] = candidate

    return risks


# ==========================================
# COLLECT ANOMALY INFORMATION
# ==========================================

def collect_anomaly_risks(
    alerts,
    start_time=None
):

    risks = {}

    for alert in alerts:

        norad_id = alert.get(
            "norad_id"
        )

        if norad_id is None:
            continue

        # Do not use future anomaly alerts
        # as if they have already occurred.

        if (
            start_time is not None
            and alert.get("timestamp")
        ):

            if (
                parse_timestamp(
                    alert["timestamp"]
                )
                > parse_timestamp(start_time)
            ):
                continue

        severity = alert.get(
            "severity",
            "NORMAL"
        )

        candidate = {

            "severity": severity,

            "suspected_type":
                alert.get(
                    "suspected_type",
                    "NORMAL"
                ),

            "anomaly_score":
                float(
                    alert.get(
                        "anomaly_score",
                        0.0
                    )
                ),

            "explanation":
                alert.get(
                    "explanation",
                    ""
                ),

            "simulated":
                bool(
                    alert.get(
                        "simulated",
                        False
                    )
                )
        }

        previous = risks.get(
            norad_id
        )

        if (
            previous is None
            or (
                severity_score(severity),
                candidate["anomaly_score"]
            ) > (
                severity_score(
                    previous["severity"]
                ),
                previous["anomaly_score"]
            )
        ):

            risks[norad_id] = candidate

    return risks


# ==========================================
# EKF UNCERTAINTY ESTIMATION
# ==========================================

def estimate_uncertainty(
    norad_id,
    start_time
):

    # Use a short historical simulated EKF
    # run ending at the planning start.
    #
    # No future observations are used.

    fusion_start = (
        start_time
        - timedelta(seconds=120)
    )

    result = run_fusion(
        norad_id=norad_id,
        duration_seconds=120,
        step_seconds=20,
        seed=42,
        start_time=fusion_start
    )

    uncertainty = float(
        result["metrics"][
            "final_uncertainty_km"
        ]
    )

    if (
        not np.isfinite(uncertainty)
        or uncertainty < 0
    ):
        raise ValueError(
            "Invalid EKF uncertainty"
        )

    return {

        "uncertainty_km":
            uncertainty,

        "source":
            "Simulated EKF covariance",

        "measurement_updates":
            int(
                result["measurement_updates"]
            ),

        "simulated":
            True
    }


# ==========================================
# EXPLAINABLE PRIORITY
# ==========================================

def explain_priority(
    conjunction_risk,
    anomaly_severity,
    uncertainty_km,
    windows
):

    reasons = []

    if conjunction_risk in [
        "CRITICAL",
        "HIGH",
        "MEDIUM"
    ]:

        reasons.append(
            f"{conjunction_risk} "
            "conjunction risk"
        )

    if anomaly_severity != "NORMAL":

        reasons.append(
            f"{anomaly_severity} "
            "anomaly detected"
        )

    if uncertainty_km is None:

        reasons.append(
            "Orbital uncertainty unavailable"
        )

    else:

        reasons.append(
            f"EKF position uncertainty "
            f"{uncertainty_km:.3f} km"
        )

    if windows:

        reasons.append(
            "Predicted sensor visibility "
            "available"
        )

    else:

        reasons.append(
            "No eligible visibility window "
            "within planning horizon"
        )

    return "; ".join(reasons)


# ==========================================
# RESOURCE CONFLICT DETECTION
# ==========================================

def has_conflict(
    sensor,
    start_time,
    end_time,
    existing_schedule,
    setup_seconds=SENSOR_SETUP_SECONDS
):

    start_time = parse_timestamp(
        start_time
    )

    end_time = parse_timestamp(
        end_time
    )

    setup = timedelta(
        seconds=setup_seconds
    )

    for task in existing_schedule:

        if task["sensor"] != sensor:
            continue

        existing_start = parse_timestamp(
            task["start_time"]
        )

        existing_end = parse_timestamp(
            task["end_time"]
        )

        if (
            start_time < existing_end + setup
            and end_time + setup > existing_start
        ):

            return True

    return False


# ==========================================
# FIND AVAILABLE SENSOR SLOT
# ==========================================

def find_available_slot(
    window,
    existing_schedule,
    deadline=None
):

    sensor = window["sensor"]

    duration = timedelta(
        seconds=SENSOR_CONFIG[
            sensor
        ]["duration_seconds"]
    )

    setup = timedelta(
        seconds=SENSOR_SETUP_SECONDS
    )

    start = parse_timestamp(
        window["start_time"]
    )

    end = parse_timestamp(
        window["end_time"]
    )

    if deadline is not None:

        end = min(
            end,
            parse_timestamp(deadline)
        )

    bookings = sorted(

        [
            task
            for task in existing_schedule
            if task["sensor"] == sensor
        ],

        key=lambda task:
            task["start_time"]
    )

    candidate = start

    for booking in bookings:

        booked_start = parse_timestamp(
            booking["start_time"]
        )

        booked_end = parse_timestamp(
            booking["end_time"]
        )

        # Booking is already finished.

        if booked_end + setup <= candidate:
            continue

        # Current gap is sufficient.

        if (
            candidate + duration + setup
            <= booked_start
        ):
            break

        # Otherwise move to the end of
        # the occupied slot.

        candidate = max(
            candidate,
            booked_end + setup
        )

    candidate_end = (
        candidate + duration
    )

    if candidate_end > end:
        return None

    if has_conflict(
        sensor,
        candidate,
        candidate_end,
        existing_schedule
    ):
        return None

    return {

        "sensor":
            sensor,

        "start_time":
            candidate.isoformat(),

        "end_time":
            candidate_end.isoformat(),

        "duration_seconds":
            int(duration.total_seconds()),

        "visibility_window_start":
            window["start_time"],

        "visibility_window_end":
            window["end_time"],

        "pass_max_elevation_deg":
            window["max_elevation_deg"],

        "pass_min_range_km":
            window["min_range_km"]
    }


# ==========================================
# GREEDY PRIORITY-BASED ALLOCATION
# ==========================================

def allocate_observations(
    candidates,
    max_tasks=30
):

    if max_tasks < 1:
        raise ValueError(
            "Maximum tasks must be positive"
        )

    schedule = []

    unscheduled = []

    ranked = sorted(

        candidates,

        key=lambda item: (
            -item["priority_score"],
            item.get("tca") or "9999",
            item["norad_id"]
        )
    )

    for candidate in ranked:

        norad_id = candidate[
            "norad_id"
        ]

        name = candidate[
            "satellite_name"
        ]

        if len(schedule) >= max_tasks:

            unscheduled.append({
                "norad_id": norad_id,
                "satellite_name": name,
                "reason":
                    "Schedule capacity reached"
            })

            continue

        windows = candidate[
            "visibility_windows"
        ]

        if not windows:

            unscheduled.append({
                "norad_id": norad_id,
                "satellite_name": name,
                "reason":
                    "No eligible visibility window"
            })

            continue

        preferences = candidate[
            "preferred_sensors"
        ]

        # High-risk conjunction observations
        # must finish before closest approach.

        deadline = None

        if candidate.get(
            "conjunction_risk"
        ) in ["CRITICAL", "HIGH"]:

            deadline = candidate.get(
                "tca"
            )

        options = []

        for window in windows:

            sensor = window["sensor"]

            if sensor not in preferences:
                continue

            slot = find_available_slot(
                window,
                schedule,
                deadline=deadline
            )

            if slot is None:
                continue

            preference_rank = (
                preferences.index(sensor)
            )

            options.append(
                (
                    slot,
                    preference_rank
                )
            )

        if not options:

            unscheduled.append({

                "norad_id":
                    norad_id,

                "satellite_name":
                    name,

                "reason": (
                    "No conflict-free preferred "
                    "sensor slot before deadline"
                    if deadline
                    else
                    "No conflict-free preferred "
                    "sensor slot"
                )
            })

            continue

        # For urgent conjunctions, choose
        # earliest feasible observation.
        #
        # Otherwise prioritise sensor suitability.

        if deadline is not None:

            options.sort(
                key=lambda item: (
                    item[0]["start_time"],
                    item[1]
                )
            )

        else:

            options.sort(
                key=lambda item: (
                    item[1],
                    item[0]["start_time"]
                )
            )

        selected = options[0][0]

        schedule.append({

            "task_id": (
                f"OBS-{norad_id}-"
                f"{len(schedule) + 1:03d}"
            ),

            "norad_id":
                norad_id,

            "satellite_name":
                name,

            **selected,

            "priority_score":
                candidate["priority_score"],

            "priority_level":
                candidate["priority_level"],

            "conjunction_risk":
                candidate["conjunction_risk"],

            "anomaly_severity":
                candidate["anomaly_severity"],

            "anomaly_type":
                candidate["anomaly_type"],

            "uncertainty_km":
                candidate["uncertainty_km"],

            "tca":
                candidate.get("tca"),

            "explanation": (
                candidate["explanation"]
                + f"; Assigned "
                f"{selected['sensor']} "
                "using conflict-free visibility"
            ),

            "status":
                "PLANNED",

            "simulated":
                True
        })

    schedule.sort(
        key=lambda task:
            task["start_time"]
    )

    return schedule, unscheduled


# ==========================================
# SCHEDULER VALIDATION
# ==========================================

def validate_schedule(schedule):

    violations = []

    for index, task in enumerate(schedule):

        start = parse_timestamp(
            task["start_time"]
        )

        end = parse_timestamp(
            task["end_time"]
        )

        window_start = parse_timestamp(
            task["visibility_window_start"]
        )

        window_end = parse_timestamp(
            task["visibility_window_end"]
        )

        if end <= start:

            violations.append(
                f"{task['task_id']}: "
                "Invalid duration"
            )

        if (
            start < window_start
            or end > window_end
        ):

            violations.append(
                f"{task['task_id']}: "
                "Outside visibility window"
            )

        if (
            task.get("conjunction_risk")
            in ["HIGH", "CRITICAL"]
            and task.get("tca")
            and end > parse_timestamp(
                task["tca"]
            )
        ):

            violations.append(
                f"{task['task_id']}: "
                "Observation ends after TCA"
            )

        remaining = schedule[
            index + 1:
        ]

        if has_conflict(
            task["sensor"],
            start,
            end,
            remaining
        ):

            violations.append(
                f"{task['task_id']}: "
                "Sensor resource conflict"
            )

    return {

        "valid":
            len(violations) == 0,

        "total_violations":
            len(violations),

        "violations":
            violations
    }


# ==========================================
# MAIN AUTONOMOUS SCHEDULER
# ==========================================

def generate_schedule(
    hours=6,
    max_objects=15,
    max_tasks=15,
    mode="demo",
    include_fusion=True,
    start_time=None
):

    if not 0 < hours <= 48:
        raise ValueError(
            "Hours must be between 0 and 48"
        )

    if not 1 <= max_objects <= 60:
        raise ValueError(
            "Invalid maximum object count"
        )

    if max_tasks < 1:
        raise ValueError(
            "Maximum tasks must be positive"
        )

    if mode not in [
        "demo",
        "catalogue"
    ]:

        raise ValueError(
            "Mode must be demo or catalogue"
        )

    if start_time is None:

        start_time = datetime.now(
            timezone.utc
        )

    start_time = parse_timestamp(
        start_time
    )

    end_time = (
        start_time
        + timedelta(hours=hours)
    )

    satellites = get_satellites()

    warnings = []

    # --------------------------------------
    # Phase 5: conjunction assessment
    # --------------------------------------

    conjunction_result = get_conjunctions(
        hours=max(1, int(np.ceil(hours))),
        step_seconds=60,
        screening_distance_km=25.0
    )

    conjunction_alerts = list(
        conjunction_result["alerts"]
    )

    anomaly_alerts = []

    # --------------------------------------
    # Phase 6: controlled demo scenario
    # --------------------------------------

    if (
        mode == "demo"
        and DEFAULT_NORAD_ID in satellites
    ):

        # Create a hypothetical encounter
        # near the first eligible radar pass,
        # when possible.

        demo_windows = predict_visibility(
            DEFAULT_NORAD_ID,
            start_time,
            hours=hours
        )

        radar_windows = [
            window
            for window in demo_windows
            if window["sensor"] == "radar"
        ]

        tca_seconds = 600

        if radar_windows:

            first_window = parse_timestamp(
                radar_windows[0]["start_time"]
            )

            tca_seconds = max(
                600,
                int(
                    (
                        first_window
                        - start_time
                    ).total_seconds()
                ) + 180
            )

        synthetic = synthetic_encounter(
            norad_id=DEFAULT_NORAD_ID,
            miss_distance_km=0.45,
            relative_speed_km_s=12.0,
            tca_seconds=tca_seconds,
            start_time=start_time
        )

        conjunction_alerts.append(
            synthetic["alert"]
        )

        # Generate historical synthetic
        # anomaly alerts, not future alerts.

        anomaly_start = (
            start_time
            - timedelta(seconds=1200)
        )

        anomaly_result = run_anomaly_demo(
            norad_id=DEFAULT_NORAD_ID,
            samples=120,
            step_seconds=10,
            strength=8.0,
            start_time=anomaly_start
        )

        anomaly_alerts.extend(
            anomaly_result["alerts"]
        )

    conjunction_risks = (
        collect_conjunction_risks(
            conjunction_alerts,
            start_time
        )
    )

    anomaly_risks = (
        collect_anomaly_risks(
            anomaly_alerts,
            start_time=start_time
        )
    )

    # --------------------------------------
    # Select candidate catalogue targets
    # --------------------------------------

    object_ids = list(
        satellites.keys()
    )

    # Risk-bearing objects are considered
    # before routine surveillance objects.

    object_ids.sort(
        key=lambda norad_id: (

            -severity_score(
                conjunction_risks.get(
                    norad_id,
                    {}
                ).get(
                    "risk_level",
                    "NORMAL"
                )
            ),

            -severity_score(
                anomaly_risks.get(
                    norad_id,
                    {}
                ).get(
                    "severity",
                    "NORMAL"
                )
            ),

            norad_id != DEFAULT_NORAD_ID,

            norad_id
        )
    )

    object_ids = object_ids[
        :max_objects
    ]

    candidates = []

    # --------------------------------------
    # Phase 2, 3 and 4 integration
    # --------------------------------------

    for norad_id in object_ids:

        metadata = satellites[
            norad_id
        ]["metadata"]

        name = metadata[
            "OBJECT_NAME"
        ]

        object_type = metadata.get(
            "COSMOTRACK_TYPE",
            "UNKNOWN"
        )

        conjunction = conjunction_risks.get(
            norad_id,
            {}
        )

        anomaly = anomaly_risks.get(
            norad_id,
            {}
        )

        conjunction_risk = conjunction.get(
            "risk_level",
            "NORMAL"
        )

        anomaly_severity = anomaly.get(
            "severity",
            "NORMAL"
        )

        anomaly_type = anomaly.get(
            "suspected_type",
            "NORMAL"
        )

        tca = conjunction.get(
            "tca"
        )

        windows = predict_visibility(
            norad_id,
            start_time,
            hours=hours
        )

        uncertainty = None

        uncertainty_source = (
            "Not estimated"
        )

        measurement_updates = None

        if include_fusion:

            try:

                fusion = estimate_uncertainty(
                    norad_id,
                    start_time
                )

                uncertainty = fusion[
                    "uncertainty_km"
                ]

                uncertainty_source = fusion[
                    "source"
                ]

                measurement_updates = fusion[
                    "measurement_updates"
                ]

            except Exception as exc:

                warnings.append(
                    f"EKF unavailable for "
                    f"{norad_id}: {exc}"
                )

        preferences = preferred_sensors(
            anomaly_type,
            conjunction_risk,
            uncertainty
        )

        eligible_windows = [
            window
            for window in windows
            if window["sensor"]
            in preferences
        ]

        visibility_score = 0.0

        if eligible_windows:

            first_window = min(
                parse_timestamp(
                    window["start_time"]
                )
                for window in eligible_windows
            )

            wait_seconds = (
                first_window - start_time
            ).total_seconds()

            # Earlier visibility receives
            # a larger priority contribution.

            visibility_score = clip_score(
                1.0
                - wait_seconds
                / (hours * 3600)
            )

        priority = calculate_priority(

            conjunction=severity_score(
                conjunction_risk
            ),

            anomaly=severity_score(
                anomaly_severity
            ),

            uncertainty=(
                calculate_uncertainty_score(
                    uncertainty
                )
            ),

            visibility=visibility_score,

            urgency=calculate_urgency(
                tca,
                start_time
            )
        )

        explanation = explain_priority(
            conjunction_risk,
            anomaly_severity,
            uncertainty,
            eligible_windows
        )

        candidates.append({

            "norad_id":
                norad_id,

            "satellite_name":
                name,

            "object_type":
                object_type,

            **priority,

            "conjunction_risk":
                conjunction_risk,

            "conjunction_simulated":
                conjunction.get(
                    "simulated",
                    False
                ),

            "miss_distance_km":
                conjunction.get(
                    "miss_distance_km"
                ),

            "tca":
                tca,

            "anomaly_severity":
                anomaly_severity,

            "anomaly_type":
                anomaly_type,

            "anomaly_simulated":
                anomaly.get(
                    "simulated",
                    False
                ),

            "uncertainty_km":
                uncertainty,

            "uncertainty_source":
                uncertainty_source,

            "measurement_updates":
                measurement_updates,

            "preferred_sensors":
                preferences,

            "visibility_windows":
                windows,

            "visibility_window_count":
                len(windows),

            "explanation":
                explanation
        })

    # --------------------------------------
    # Allocate sensor resources
    # --------------------------------------

    candidates.sort(
        key=lambda item:
            -item["priority_score"]
    )

    schedule, unscheduled = (
        allocate_observations(
            candidates,
            max_tasks=max_tasks
        )
    )

    validation = validate_schedule(
        schedule
    )

    # --------------------------------------
    # Calculate scheduler metrics
    # --------------------------------------

    sensor_utilisation = {}

    for sensor in SENSOR_CONFIG:

        tasks = [
            task
            for task in schedule
            if task["sensor"] == sensor
        ]

        allocated_seconds = sum(
            task["duration_seconds"]
            for task in tasks
        )

        sensor_utilisation[sensor] = {

            "tasks":
                len(tasks),

            "allocated_seconds":
                allocated_seconds,

            "utilisation_percent":
                round(
                    allocated_seconds
                    / (hours * 3600)
                    * 100,
                    4
                )
        }

    priority_counts = {
        "CRITICAL": 0,
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0
    }

    for candidate in candidates:

        priority_counts[
            candidate["priority_level"]
        ] += 1

    urgent_targets = [

        candidate

        for candidate in candidates

        if candidate["conjunction_risk"]
        in ["CRITICAL", "HIGH"]
        or candidate["anomaly_severity"]
        in ["CRITICAL", "HIGH"]
    ]

    scheduled_ids = {
        task["norad_id"]
        for task in schedule
    }

    urgent_scheduled = sum(

        candidate["norad_id"]
        in scheduled_ids

        for candidate in urgent_targets
    )

    return {

        "mode":
            mode,

        "algorithm":
            "Weighted priority + greedy resource allocation",

        "simulated":
            True,

        "station":
            "Bengaluru Ground Station",

        "planning_start":
            start_time.isoformat(),

        "planning_end":
            end_time.isoformat(),

        "planning_hours":
            hours,

        "total_catalogue_objects":
            len(satellites),

        "total_candidates":
            len(candidates),

        "total_scheduled":
            len(schedule),

        "total_unscheduled":
            len(unscheduled),

        "conjunction_alert_count":
            len(conjunction_alerts),

        "anomaly_alert_count":
            len(anomaly_alerts),

        "priority_weights":
            PRIORITY_WEIGHTS,

        "priority_counts":
            priority_counts,

        "urgent_targets":
            len(urgent_targets),

        "urgent_targets_scheduled":
            urgent_scheduled,

        "urgent_coverage_percent": (
            round(
                urgent_scheduled
                / len(urgent_targets)
                * 100,
                2
            )
            if urgent_targets
            else None
        ),

        "schedule_success_percent":
            round(
                len(schedule)
                / max(len(candidates), 1)
                * 100,
                2
            ),

        "sensor_utilisation":
            sensor_utilisation,

        "validation":
            validation,

        "warnings":
            warnings,

        "candidates":
            candidates,

        "schedule":
            schedule,

        "unscheduled":
            unscheduled
    }


# ==========================================
# MODULE TEST
# ==========================================

if __name__ == "__main__":

    import json

    print("=" * 65)

    print(
        "COSMOTRACK AUTONOMOUS "
        "OBSERVATION SCHEDULER"
    )

    print("=" * 65)

    result = generate_schedule(
        hours=6,
        max_objects=10,
        max_tasks=10,
        mode="demo"
    )

    print(
        "\nPlanning start:",
        result["planning_start"]
    )

    print(
        "Planning end:",
        result["planning_end"]
    )

    print(
        "Total candidates:",
        result["total_candidates"]
    )

    print(
        "Scheduled:",
        result["total_scheduled"]
    )

    print(
        "Unscheduled:",
        result["total_unscheduled"]
    )

    print(
        "Schedule valid:",
        result["validation"]["valid"]
    )

    print(
        "Urgent target coverage:",
        result["urgent_coverage_percent"]
    )

    print("\nPRIORITY QUEUE")

    print("-" * 65)

    for candidate in result["candidates"]:

        print(
            f"{candidate['satellite_name']:<25}"
            f"{candidate['priority_score']:>8.2f}"
            f" | {candidate['priority_level']}"
        )

    print("\nOBSERVATION SCHEDULE")

    print("-" * 65)

    for task in result["schedule"]:

        print(
            task["start_time"],
            "|",
            task["sensor"].upper(),
            "|",
            task["satellite_name"],
            "|",
            task["priority_level"]
        )

    print("\nSENSOR UTILISATION")

    print(json.dumps(
        result["sensor_utilisation"],
        indent=2
    ))

    print("\nVALIDATION")

    print(json.dumps(
        result["validation"],
        indent=2
    ))

    if result["warnings"]:

        print("\nWARNINGS")

        for warning in result["warnings"]:

            print(warning)

    if result["schedule"]:

        print("\nSAMPLE EXPLAINABLE TASK")

        print(json.dumps(
            result["schedule"][0],
            indent=2
        ))

    print(
        "\nAutonomous scheduling completed!"
    )
