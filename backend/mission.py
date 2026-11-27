
from datetime import datetime, timezone
from time import perf_counter

from backend.orbit import (
    get_satellites,
    get_state,
    list_satellites
)

from backend.collision import (
    get_conjunctions
)

from backend.anomaly import (
    run_anomaly_demo,
    evaluate_model,
    MODEL_NAME
)

from backend.fusion import (
    run_fusion
)

from backend.scheduler import (
    generate_schedule
)

from backend.cache import (
    CONJUNCTION_CACHE,
    ANOMALY_CACHE,
    FUSION_CACHE,
    SCHEDULER_CACHE,
    MISSION_CACHE,
    EVALUATION_CACHE,
    get_cache_stats
)


# ==========================================
# CONFIGURATION
# ==========================================

APP_NAME = "CosmoTrack"

APP_VERSION = "1.0.0"

DEFAULT_NORAD_ID = 25544

SEVERITY_PRIORITY = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
    "NORMAL": 0
}


# ==========================================
# TIMESTAMP UTILITIES
# ==========================================

def utc_now():

    return datetime.now(
        timezone.utc
    ).isoformat()


def parse_timestamp(value):

    if value is None:
        return None

    result = datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )

    if result.tzinfo is None:
        result = result.replace(
            tzinfo=timezone.utc
        )

    return result.astimezone(
        timezone.utc
    )


# ==========================================
# SAFE MODULE EXECUTION
# ==========================================

def safe_execute(
    module_name,
    function,
    module_status,
    warnings
):

    started = perf_counter()

    try:

        result = function()

        module_status[module_name] = {
            "status": "AVAILABLE",

            "response_time_ms": round(
                (perf_counter() - started)
                * 1000,
                2
            )
        }

        return result

    except Exception as exc:

        module_status[module_name] = {
            "status": "DEGRADED",

            "response_time_ms": round(
                (perf_counter() - started)
                * 1000,
                2
            ),

            "error_type":
                type(exc).__name__
        }

        warnings.append(
            f"{module_name} unavailable: "
            f"{type(exc).__name__}"
        )

        return None


# ==========================================
# CACHED MODULE ACCESS
# ==========================================

def mission_conjunctions(hours=6):

    return CONJUNCTION_CACHE.get_or_create(

        ("conjunctions", hours),

        lambda: get_conjunctions(
            hours=hours,
            step_seconds=60,
            screening_distance_km=25.0
        )
    )


def mission_anomalies(norad_id):

    return ANOMALY_CACHE.get_or_create(

        ("anomalies", norad_id),

        lambda: run_anomaly_demo(
            norad_id=norad_id,
            samples=120,
            step_seconds=10,
            strength=8.0
        )
    )


def mission_fusion(norad_id):

    return FUSION_CACHE.get_or_create(

        ("fusion", norad_id),

        lambda: run_fusion(
            norad_id=norad_id,
            duration_seconds=300,
            step_seconds=10,
            seed=42
        )
    )


def mission_evaluation():

    return EVALUATION_CACHE.get_or_create(

        "default_evaluation",

        lambda: evaluate_model(
            samples=500,
            anomaly_fraction=0.10,
            strength=8.0
        )
    )


def mission_schedule(
    hours=6,
    max_objects=10,
    mode="demo",
    include_fusion=True
):

    key = (
        hours,
        max_objects,
        mode,
        include_fusion
    )

    return SCHEDULER_CACHE.get_or_create(

        key,

        lambda: generate_schedule(
            hours=hours,
            max_objects=max_objects,
            max_tasks=max_objects,
            mode=mode,
            include_fusion=include_fusion
        )
    )


# ==========================================
# CATALOGUE SUMMARY
# ==========================================

def catalogue_summary():

    objects = list_satellites()

    counts = {
        "STATION": 0,
        "SATELLITE": 0,
        "DEBRIS": 0,
        "UNKNOWN": 0
    }

    for item in objects:

        object_type = item.get(
            "type",
            "UNKNOWN"
        )

        if object_type not in counts:
            object_type = "UNKNOWN"

        counts[object_type] += 1

    return {

        "total_objects":
            len(objects),

        "space_stations":
            counts["STATION"],

        "active_satellites":
            counts["SATELLITE"],

        "debris_objects":
            counts["DEBRIS"],

        "unknown_objects":
            counts["UNKNOWN"],

        "source":
            "CelesTrak orbital catalogue",

        "propagation_method":
            "SGP4 / Skyfield",

        "catalogue_scope":
            "Selected LEO catalogue subset"
    }


# ==========================================
# NORMALISE CONJUNCTION ALERT
# ==========================================

def normalise_conjunction_alert(alert):

    severity = alert.get(
        "risk_level",
        "LOW"
    )

    object_a = alert.get(
        "object_a",
        {}
    )

    object_b = alert.get(
        "object_b",
        {}
    )

    name_a = object_a.get(
        "name",
        object_a.get(
            "satellite_name",
            "Object A"
        )
    )

    name_b = object_b.get(
        "name",
        object_b.get(
            "satellite_name",
            "Object B"
        )
    )

    return {

        "alert_id":
            str(alert["alert_id"]),

        "category":
            "CONJUNCTION",

        "severity":
            severity,

        "priority":
            SEVERITY_PRIORITY.get(
                severity,
                0
            ),

        "title":
            f"Close approach: {name_a} / {name_b}",

        "description":
            alert.get("explanation", ""),

        "timestamp":
            alert.get("tca"),

        "timestamp_type":
            "PREDICTED_CLOSEST_APPROACH",

        "recommended_action":
            alert.get(
                "recommended_action",
                "Review conjunction assessment"
            ),

        "norad_id":
            object_a.get("norad_id"),

        "simulated":
            bool(
                alert.get("simulated", False)
            ),

        "source":
            "Conjunction assessment engine",

        "details": {
            "object_a": object_a,
            "object_b": object_b,

            "miss_distance_km":
                alert.get("miss_distance_km"),

            "relative_velocity_km_s":
                alert.get(
                    "relative_velocity_km_s"
                ),

            "tca":
                alert.get("tca"),

            "collision_probability":
                alert.get(
                    "collision_probability"
                )
        }
    }


# ==========================================
# NORMALISE ANOMALY ALERT
# ==========================================

def normalise_anomaly_alert(alert):

    severity = alert.get(
        "severity",
        "LOW"
    )

    anomaly_type = alert.get(
        "suspected_type",
        "UNKNOWN"
    )

    return {

        "alert_id":
            str(alert["alert_id"]),

        "category":
            "ANOMALY",

        "severity":
            severity,

        "priority":
            SEVERITY_PRIORITY.get(
                severity,
                0
            ),

        "title": (
            anomaly_type.replace("_", " ").title()
            + " — "
            + alert.get(
                "satellite_name",
                "Unknown satellite"
            )
        ),

        "description":
            alert.get("explanation", ""),

        "timestamp":
            alert.get("timestamp"),

        "timestamp_type":
            "SIMULATED_OBSERVATION",

        "recommended_action":
            alert.get(
                "recommended_action",
                "Review anomaly"
            ),

        "norad_id":
            alert.get("norad_id"),

        "simulated":
            True,

        "source":
            "Hybrid AI anomaly engine",

        "details": {

            "anomaly_type":
                anomaly_type,

            "anomaly_score":
                alert.get("anomaly_score"),

            "detection_source":
                alert.get("detection_source"),

            "contributors":
                alert.get("contributors", []),

            "position_km":
                alert.get("position_km"),

            "score_is_probability":
                False
        }
    }


# ==========================================
# CENTRAL ALERT FEED
# ==========================================

def build_alert_feed(
    conjunction_alerts=None,
    anomaly_alerts=None,
    severity=None,
    category=None,
    limit=100
):

    if limit < 1:
        raise ValueError(
            "Alert limit must be positive"
        )

    conjunction_alerts = (
        conjunction_alerts or []
    )

    anomaly_alerts = (
        anomaly_alerts or []
    )

    alerts = [

        normalise_conjunction_alert(alert)

        for alert in conjunction_alerts
    ]

    alerts.extend([

        normalise_anomaly_alert(alert)

        for alert in anomaly_alerts
    ])

    # Remove duplicate alerts within
    # the same category.

    unique = {}

    for alert in alerts:

        key = (
            alert["category"],
            alert["alert_id"]
        )

        unique[key] = alert

    alerts = list(unique.values())

    if severity is not None:

        alerts = [

            alert

            for alert in alerts

            if alert["severity"]
            == severity.upper()
        ]

    if category is not None:

        alerts = [

            alert

            for alert in alerts

            if alert["category"]
            == category.upper()
        ]

    # Critical alerts first.
    # Conjunctions: earliest TCA first.
    # Anomalies: latest timestamp first.

    def alert_sort_key(alert):

        timestamp = parse_timestamp(
            alert["timestamp"]
        )

        seconds = (
            timestamp.timestamp()
            if timestamp is not None
            else 0
        )

        if alert["category"] == "ANOMALY":
            seconds = -seconds

        return (
            -alert["priority"],
            alert["category"],
            seconds
        )

    alerts.sort(key=alert_sort_key)

    counts = {
        "CRITICAL": 0,
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0,
        "NORMAL": 0
    }

    for alert in alerts:

        level = alert["severity"]

        if level in counts:
            counts[level] += 1

    return {

        "total_alerts":
            len(alerts),

        "returned_alerts":
            min(len(alerts), limit),

        "severity_counts":
            counts,

        "conjunction_alerts":
            sum(
                alert["category"] == "CONJUNCTION"
                for alert in alerts
            ),

        "anomaly_alerts":
            sum(
                alert["category"] == "ANOMALY"
                for alert in alerts
            ),

        "simulated_alerts":
            sum(
                alert["simulated"]
                for alert in alerts
            ),

        "alerts":
            alerts[:limit]
    }


# ==========================================
# SUMMARISE INDIVIDUAL MODULES
# ==========================================

def summarise_fusion(result):

    if result is None:
        return None

    return {

        "algorithm":
            result.get("algorithm"),

        "simulated":
            True,

        "measurement_updates":
            result.get("measurement_updates"),

        "total_samples":
            result.get("total_samples"),

        **result["metrics"]
    }


def summarise_anomalies(
    result,
    evaluation
):

    if result is None and evaluation is None:
        return None

    summary = {

        "model":
            MODEL_NAME,

        "simulated":
            True,

        "evaluation":
            evaluation
    }

    if result is not None:

        summary.update({

            "total_samples":
                result["total_samples"],

            "injected_anomalies":
                result["injected_anomalies"],

            "detected_anomalies":
                result["detected_anomalies"],

            "severity_counts":
                result["severity_counts"],

            "detection_source_counts":
                result.get(
                    "detection_source_counts",
                    {}
                )
        })

    return summary


def summarise_scheduler(result):

    if result is None:
        return None

    keys = [

        "algorithm",
        "mode",
        "simulated",
        "planning_start",
        "planning_end",
        "total_candidates",
        "total_scheduled",
        "total_unscheduled",
        "priority_counts",
        "urgent_coverage_percent",
        "schedule_success_percent",
        "sensor_utilisation",
        "validation",
        "warnings"
    ]

    summary = {

        key: result.get(key)

        for key in keys
    }

    summary["next_observation"] = (

        result["schedule"][0]

        if result["schedule"]
        else None
    )

    summary["highest_priority_target"] = (

        result["candidates"][0]

        if result["candidates"]
        else None
    )

    return summary


# ==========================================
# UNIFIED MISSION OVERVIEW
# ==========================================

def build_mission_overview(
    norad_id=DEFAULT_NORAD_ID,
    hours=6,
    max_objects=10,
    mode="demo",
    include_fusion=True
):

    if mode not in ["demo", "catalogue"]:
        raise ValueError("Invalid mission mode")

    if not 1 <= hours <= 24:
        raise ValueError("Invalid planning horizon")

    if not 1 <= max_objects <= 60:
        raise ValueError("Invalid object count")

    started = perf_counter()

    satellites = get_satellites()

    if norad_id not in satellites:

        raise KeyError(
            f"Unknown satellite: {norad_id}"
        )

    module_status = {}

    warnings = []

    # --------------------------------------
    # Orbital catalogue
    # --------------------------------------

    catalogue = safe_execute(

        "catalogue",

        catalogue_summary,

        module_status,

        warnings
    )

    # --------------------------------------
    # Selected satellite
    # --------------------------------------

    selected_satellite = safe_execute(

        "orbit",

        lambda: get_state(norad_id),

        module_status,

        warnings
    )

    # --------------------------------------
    # Conjunction assessment
    # --------------------------------------

    conjunctions = safe_execute(

        "conjunctions",

        lambda: mission_conjunctions(hours),

        module_status,

        warnings
    )

    # --------------------------------------
    # EKF sensor fusion
    # --------------------------------------

    fusion = None

    if include_fusion:

        fusion = safe_execute(

            "fusion",

            lambda: mission_fusion(norad_id),

            module_status,

            warnings
        )

    else:

        module_status["fusion"] = {
            "status": "DISABLED"
        }

    # --------------------------------------
    # Anomaly detection
    # --------------------------------------

    anomalies = None

    evaluation = None

    if mode == "demo":

        anomalies = safe_execute(

            "anomalies",

            lambda: mission_anomalies(
                norad_id
            ),

            module_status,

            warnings
        )

        evaluation = safe_execute(

            "ml_evaluation",

            mission_evaluation,

            module_status,

            warnings
        )

    else:

        module_status["anomalies"] = {
            "status": "DISABLED",
            "reason": (
                "No live anomaly observation feed"
            )
        }

    # --------------------------------------
    # Autonomous sensor scheduling
    # --------------------------------------

    scheduler = safe_execute(

        "scheduler",

        lambda: mission_schedule(

            hours=hours,

            max_objects=max_objects,

            mode=mode,

            include_fusion=include_fusion
        ),

        module_status,

        warnings
    )

    # --------------------------------------
    # Central alert feed
    # --------------------------------------

    alert_feed = build_alert_feed(

        conjunction_alerts=(
            conjunctions["alerts"]
            if conjunctions is not None
            else []
        ),

        anomaly_alerts=(
            anomalies["alerts"]
            if anomalies is not None
            else []
        ),

        limit=100
    )

    # --------------------------------------
    # Module health
    # --------------------------------------

    degraded = any(

        item["status"] == "DEGRADED"

        for item in module_status.values()
    )

    return {

        "application":
            APP_NAME,

        "version":
            APP_VERSION,

        "generated_at":
            utc_now(),

        "mode":
            mode,

        "system_status": (
            "DEGRADED"
            if degraded
            else "OPERATIONAL"
        ),

        "selected_norad_id":
            norad_id,

        "planning_hours":
            hours,

        "catalogue":
            catalogue,

        "selected_satellite":
            selected_satellite,

        "conjunctions": (

            {

                "total_alerts":
                    conjunctions["total_alerts"],

                "risk_counts":
                    conjunctions["risk_counts"],

                "screened_pairs":
                    conjunctions["screened_pairs"],

                "screening_start":
                    conjunctions.get(
                        "screening_start"
                    ),

                "screening_end":
                    conjunctions.get(
                        "screening_end"
                    ),

                "collision_probability_available":
                    False
            }

            if conjunctions is not None
            else None
        ),

        "fusion":
            summarise_fusion(fusion),

        "anomalies":
            summarise_anomalies(
                anomalies,
                evaluation
            ),

        "scheduler":
            summarise_scheduler(scheduler),

        "alert_feed":
            alert_feed,

        "module_status":
            module_status,

        "warnings":
            warnings,

        "data_provenance": {

            "orbital_elements":
                "Real CelesTrak catalogue data",

            "orbital_positions":
                "SGP4 model predictions",

            "sensor_measurements":
                "Simulated",

            "fusion_estimates":
                "Simulated sensor-based EKF",

            "anomaly_training":
                "Synthetic residuals",

            "conjunction_assessment":
                "Catalogue-based distance screening",

            "sensor_scheduling":
                "Simulated observation planning",

            "synthetic_scenarios_enabled":
                mode == "demo"
        },

        "response_time_ms": round(

            (perf_counter() - started)
            * 1000,

            2
        )
    }


# ==========================================
# CACHED MISSION OVERVIEW
# ==========================================

def get_mission_overview(
    norad_id=DEFAULT_NORAD_ID,
    hours=6,
    max_objects=10,
    mode="demo",
    include_fusion=True
):

    key = (
        norad_id,
        hours,
        max_objects,
        mode,
        include_fusion
    )

    started = perf_counter()

    result = MISSION_CACHE.get_or_create(

        key,

        lambda: build_mission_overview(

            norad_id=norad_id,

            hours=hours,

            max_objects=max_objects,

            mode=mode,

            include_fusion=include_fusion
        )
    )

    result["request_time_ms"] = round(

        (perf_counter() - started)
        * 1000,

        2
    )

    return result


# ==========================================
# INDEPENDENT CENTRAL ALERT ENDPOINT
# ==========================================

def get_mission_alerts(
    norad_id=DEFAULT_NORAD_ID,
    hours=6,
    mode="demo",
    severity=None,
    category=None,
    limit=100
):

    if mode not in ["demo", "catalogue"]:
        raise ValueError("Invalid mission mode")

    if norad_id not in get_satellites():
        raise KeyError("Unknown satellite")

    conjunctions = mission_conjunctions(
        hours
    )

    anomalies = None

    if mode == "demo":

        anomalies = mission_anomalies(
            norad_id
        )

    return {

        "generated_at":
            utc_now(),

        "mode":
            mode,

        **build_alert_feed(

            conjunction_alerts=(
                conjunctions["alerts"]
            ),

            anomaly_alerts=(
                anomalies["alerts"]
                if anomalies is not None
                else []
            ),

            severity=severity,

            category=category,

            limit=limit
        )
    }


# ==========================================
# SYSTEM INFORMATION
# ==========================================

def get_system_info():

    return {

        "application":
            APP_NAME,

        "version":
            APP_VERSION,

        "timestamp":
            utc_now(),

        "modules": [

            "Orbital tracking",

            "Multi-sensor simulation",

            "Extended Kalman Filter",

            "Conjunction assessment",

            "Hybrid AI anomaly detection",

            "Autonomous sensor scheduling",

            "Unified mission intelligence"
        ],

        "cache":
            get_cache_stats(),

        "hardware_connected":
            False,

        "deployment_type":
            "Research and competition prototype"
    }


# ==========================================
# MODULE TEST
# ==========================================

if __name__ == "__main__":

    import json

    print("=" * 65)

    print("COSMOTRACK MISSION CONTROL")

    print("=" * 65)

    result = get_mission_overview(

        hours=6,

        max_objects=5,

        mode="demo"
    )

    print(json.dumps(
        result,
        indent=2
    ))

    print("\nCACHE PERFORMANCE TEST")

    cached = get_mission_overview(

        hours=6,

        max_objects=5,

        mode="demo"
    )

    print(
        "Initial request:",
        result["request_time_ms"],
        "ms"
    )

    print(
        "Repeated request:",
        cached["request_time_ms"],
        "ms"
    )

    print("\nCACHE STATISTICS")

    print(json.dumps(
        get_cache_stats(),
        indent=2
    ))
