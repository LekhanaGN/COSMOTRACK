
from fastapi import FastAPI, HTTPException, Query
from backend.sensors import (
    simulate_sensor_series,
    STATION,
    SENSOR_NOISE
)
from backend.orbit import (
    list_satellites,
    get_state,
    get_trajectory
)
from backend.fusion import run_fusion
from backend.collision import (
    get_conjunctions,
    synthetic_encounter,
    RISK_THRESHOLDS
)
from backend.anomaly import (
    analyze_observation,
    evaluate_model,
    run_anomaly_demo,
    FEATURE_NAMES,
    NORMAL_SIGMA
)
from backend.scheduler import (
    generate_schedule,
    predict_visibility,
    calculate_priority,
    PRIORITY_WEIGHTS,
    SENSOR_CONFIG,
    SENSOR_SETUP_SECONDS,
    UNCERTAINTY_REFERENCE_KM
)

from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field

app = FastAPI(
    title="CosmoTrack API",
    description="AI-Powered Space Domain Awareness",
    version="0.2.0"
)


@app.get("/")
def root():
    return {
        "project": "CosmoTrack",
        "status": "online",
        "mode": "prototype"
    }


@app.get("/api/health")
def health():
    return {"status": "healthy"}


@app.get("/api/satellites")
def satellites():
    objects = list_satellites()

    return {
        "total": len(objects),
        "objects": objects
    }


@app.get("/api/satellites/{norad_id}")
def satellite_state(norad_id: int):
    try:
        return get_state(norad_id)

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Satellite not found"
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc


@app.get("/api/orbits/{norad_id}")
def satellite_orbit(
    norad_id: int,
    minutes: int = Query(100, ge=1, le=1440),
    step: int = Query(60, ge=30, le=600)
):
    try:
        return get_trajectory(
            norad_id,
            duration_minutes=minutes,
            step_seconds=step
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Satellite not found"
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc


@app.get("/api/summary")
def summary():
    objects = list_satellites()

    return {
        "total_objects": len(objects),
        "satellites": sum(
            obj["type"] == "SATELLITE"
            for obj in objects
        ),
        "stations": sum(
            obj["type"] == "STATION"
            for obj in objects
        ),
        "debris": sum(
            obj["type"] == "DEBRIS"
            for obj in objects
        ),
        "source": "CelesTrak",
        "propagation": "SGP4",
        "mode": "prototype"
    }

@app.get("/api/sensors/config")
def sensor_configuration():

    return {
        "station": STATION,
        "noise_assumptions": SENSOR_NOISE,
        "available_sensors": [
            "lidar",
            "radar",
            "optical"
        ],
        "mode": "simulation"
    }


@app.get("/api/sensors/{norad_id}")
def sensor_data(
    norad_id: int,
    duration: int = Query(
        300, ge=30, le=900
    ),
    step: int = Query(
        10, ge=5, le=60
    ),
    noise: float = Query(
        1.0, ge=0.1, le=10.0
    ),
    dropout: str | None = None,
    seed: int = Query(42, ge=0)
):

    allowed = [
        None, "lidar", "radar", "optical"
    ]

    if dropout not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Invalid dropout sensor"
        )

    try:

        return simulate_sensor_series(
            norad_id=norad_id,
            duration_seconds=duration,
            step_seconds=step,
            noise_multiplier=noise,
            dropout_sensor=dropout,
            seed=seed
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Satellite not found"
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

@app.get("/api/fusion/{norad_id}")
def fusion_data(
    norad_id: int,

    duration: int = Query(
        300, ge=30, le=900
    ),

    step: int = Query(
        10, ge=5, le=60
    ),

    noise: float = Query(
        1.0, ge=0.1, le=10.0
    ),

    dropout: str | None = None,

    seed: int = Query(
        42, ge=0
    )
):

    allowed = [
        None,
        "lidar",
        "radar",
        "optical"
    ]

    if dropout not in allowed:

        raise HTTPException(
            status_code=400,
            detail="Invalid dropout sensor"
        )

    try:

        return run_fusion(
            norad_id=norad_id,
            duration_seconds=duration,
            step_seconds=step,
            noise_multiplier=noise,
            dropout_sensor=dropout,
            seed=seed
        )

    except KeyError as exc:

        raise HTTPException(
            status_code=404,
            detail="Satellite not found"
        ) from exc

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc


@app.get("/api/fusion/{norad_id}/metrics")
def fusion_metrics(norad_id: int):

    try:

        result = run_fusion(norad_id)

        return {
            "id": norad_id,
            "name": result["name"],
            "algorithm": result["algorithm"],
            "simulated": True,
            "metrics": result["metrics"],
            "measurement_updates":
                result["measurement_updates"]
        }

    except KeyError as exc:

        raise HTTPException(
            status_code=404,
            detail="Satellite not found"
        ) from exc

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

# ==========================================
# CONJUNCTION CONFIGURATION
# ==========================================

@app.get("/api/conjunctions/config")
def conjunction_config():

    return {
        "risk_thresholds_km": RISK_THRESHOLDS,

        "default_prediction_hours": 24,

        "default_step_seconds": 60,

        "default_screening_distance_km": 25,

        "collision_probability_available":
            False,

        "classification_method":
            "Distance-based screening priority",

        "mode": "prototype"
    }


# ==========================================
# REAL CATALOGUE CONJUNCTION SCREENING
# ==========================================

@app.get("/api/conjunctions")
def conjunction_screening(

    hours: int = Query(
        24, ge=1, le=48
    ),

    step: int = Query(
        60, ge=15, le=120
    ),

    threshold: float = Query(
        25.0, ge=1.0, le=100.0
    )
):

    try:

        return get_conjunctions(
            hours=hours,
            step_seconds=step,
            screening_distance_km=threshold
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc


# ==========================================
# SYNTHETIC ENCOUNTER DEMONSTRATION
# ==========================================

@app.get("/api/conjunctions/demo")
def conjunction_demo(

    norad_id: int = 25544,

    miss_distance: float = Query(
        0.45, ge=0.01, le=100
    ),

    relative_speed: float = Query(
        12.0, ge=0.1, le=20.0
    ),

    tca_seconds: int = Query(
        600, ge=60, le=3600
    )
):

    try:

        return synthetic_encounter(
            norad_id=norad_id,

            miss_distance_km=miss_distance,

            relative_speed_km_s=relative_speed,

            tca_seconds=tca_seconds
        )

    except KeyError as exc:

        raise HTTPException(
            status_code=404,
            detail="Satellite not found"
        ) from exc

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc


# ==========================================
# CONJUNCTION SUMMARY
# ==========================================

@app.get("/api/conjunctions/summary")
def conjunction_summary():

    result = get_conjunctions()

    return {
        "screening_start":
            result["screening_start"],

        "screening_end":
            result["screening_end"],

        "total_objects":
            result["total_objects"],

        "screened_pairs":
            result["screened_pairs"],

        "total_alerts":
            result["total_alerts"],

        "risk_counts":
            result["risk_counts"],

        "highest_priority_alert": (
            result["alerts"][0]
            if result["alerts"]
            else None
        ),

        "mode": "catalogue_prediction"
    }

# ==========================================
# ANOMALY CONFIGURATION
# ==========================================

@app.get("/api/anomalies/config")
def anomaly_config():

    return {
        "model": "Isolation Forest",

        "learning_type":
            "Unsupervised machine learning",

        "training_samples": 2000,

        "n_estimators": 200,

        "contamination": 0.02,

        "features": [
            {
                "name": name,

                "normal_noise_scale":
                    float(sigma)
            }

            for name, sigma in zip(
                FEATURE_NAMES,
                NORMAL_SIGMA
            )
        ],

        "decision_threshold": 0,

        "score_is_probability": False,

        "training_source":
            "Simulated normal residuals"
    }


# ==========================================
# MODEL PERFORMANCE
# ==========================================

@app.get("/api/anomalies/metrics")
def anomaly_metrics(

    samples: int = Query(
        500, ge=100, le=5000
    ),

    anomaly_fraction: float = Query(
        0.10, ge=0.01, le=0.50
    ),

    strength: float = Query(
        8.0, ge=1.0, le=20.0
    )
):

    return evaluate_model(
        samples=samples,
        anomaly_fraction=anomaly_fraction,
        strength=strength
    )


# ==========================================
# SATELLITE ANOMALY DEMO
# ==========================================

@app.get("/api/anomalies/demo")
def anomaly_demo(

    norad_id: int = 25544,

    samples: int = Query(
        120, ge=30, le=500
    ),

    step_seconds: int = Query(
        10, ge=1, le=60
    ),

    strength: float = Query(
        8.0, ge=1.0, le=20.0
    )
):

    try:

        return run_anomaly_demo(
            norad_id=norad_id,
            samples=samples,
            step_seconds=step_seconds,
            strength=strength
        )

    except KeyError as exc:

        raise HTTPException(
            status_code=404,
            detail="Satellite not found"
        ) from exc

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc


# ==========================================
# ANALYZE CUSTOM RESIDUAL OBSERVATION
# ==========================================

class AnomalyObservation(BaseModel):

    position_residual_m: float = Field(
        ge=0
    )

    velocity_residual_m_s: float = Field(
        ge=0
    )

    lidar_residual_m: float

    radar_residual_m: float

    radar_range_rate_residual_m_s: float

    optical_residual_deg: float = Field(
        ge=0
    )


@app.post("/api/anomalies/analyze")
def anomaly_analyze(
    observation: AnomalyObservation
):

    try:

        features = [
            getattr(observation, name)
            for name in FEATURE_NAMES
        ]

        return analyze_observation(
            features
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

# ==========================================
# SCHEDULER RESPONSE CACHE
# ==========================================

@lru_cache(maxsize=8)
def cached_schedule(
    hours,
    max_objects,
    max_tasks,
    mode,
    include_fusion,
    time_bucket
):

    return generate_schedule(

        hours=hours,

        max_objects=max_objects,

        max_tasks=max_tasks,

        mode=mode,

        include_fusion=include_fusion,

        start_time=datetime.fromtimestamp(
            time_bucket * 300,
            tz=timezone.utc
        )
    )


def get_cached_schedule(
    hours=6,
    max_objects=10,
    max_tasks=10,
    mode="demo",
    include_fusion=True
):

    # Recompute schedules every five minutes
    # to avoid repeated expensive EKF runs.

    time_bucket = int(
        datetime.now(
            timezone.utc
        ).timestamp() // 300
    )

    return cached_schedule(

        hours,

        max_objects,

        max_tasks,

        mode,

        include_fusion,

        time_bucket
    )


# ==========================================
# SCHEDULER CONFIGURATION
# ==========================================

@app.get("/api/scheduler/config")
def scheduler_config():

    return {

        "algorithm":
            "Weighted priority + greedy allocation",

        "priority_weights":
            PRIORITY_WEIGHTS,

        "sensors":
            SENSOR_CONFIG,

        "sensor_setup_seconds":
            SENSOR_SETUP_SECONDS,

        "uncertainty_reference_km":
            UNCERTAINTY_REFERENCE_KM,

        "priority_range":
            [0, 100],

        "available_modes": [
            "demo",
            "catalogue"
        ],

        "simulated":
            True,

        "cache_duration_seconds":
            300
    }


# ==========================================
# GENERATE AUTONOMOUS SCHEDULE
# ==========================================

@app.get("/api/scheduler")
def scheduler_plan(

    hours: int = Query(
        6, ge=1, le=24
    ),

    max_objects: int = Query(
        10, ge=1, le=60
    ),

    max_tasks: int = Query(
        10, ge=1, le=60
    ),

    mode: Literal[
        "demo",
        "catalogue"
    ] = "demo",

    include_fusion: bool = True
):

    return get_cached_schedule(

        hours=hours,

        max_objects=max_objects,

        max_tasks=max_tasks,

        mode=mode,

        include_fusion=include_fusion
    )


# ==========================================
# SCHEDULER SUMMARY
# ==========================================

@app.get("/api/scheduler/summary")
def scheduler_summary():

    result = get_cached_schedule()

    return {

        "algorithm":
            result["algorithm"],

        "mode":
            result["mode"],

        "simulated":
            result["simulated"],

        "planning_start":
            result["planning_start"],

        "planning_end":
            result["planning_end"],

        "total_candidates":
            result["total_candidates"],

        "total_scheduled":
            result["total_scheduled"],

        "total_unscheduled":
            result["total_unscheduled"],

        "priority_counts":
            result["priority_counts"],

        "urgent_coverage_percent":
            result["urgent_coverage_percent"],

        "schedule_success_percent":
            result["schedule_success_percent"],

        "sensor_utilisation":
            result["sensor_utilisation"],

        "validation":
            result["validation"],

        "warnings":
            result["warnings"],

        "highest_priority_target": (
            result["candidates"][0]
            if result["candidates"]
            else None
        ),

        "next_observation": (
            result["schedule"][0]
            if result["schedule"]
            else None
        )
    }


# ==========================================
# INDIVIDUAL SATELLITE VISIBILITY
# ==========================================

@app.get(
    "/api/scheduler/visibility/{norad_id}"
)
def scheduler_visibility(

    norad_id: int,

    hours: int = Query(
        6, ge=1, le=24
    )
):

    try:

        start_time = datetime.now(
            timezone.utc
        )

        windows = predict_visibility(

            norad_id,

            start_time,

            hours=hours
        )

        return {

            "norad_id":
                norad_id,

            "planning_start":
                start_time.isoformat(),

            "planning_hours":
                hours,

            "total_windows":
                len(windows),

            "windows":
                windows,

            "simulated_sensor_limits":
                True
        }

    except KeyError as exc:

        raise HTTPException(

            status_code=404,

            detail="Satellite not found"

        ) from exc


# ==========================================
# CUSTOM PRIORITY CALCULATION
# ==========================================

class PriorityInput(BaseModel):

    conjunction: float = Field(
        default=0.0,
        ge=0,
        le=1
    )

    anomaly: float = Field(
        default=0.0,
        ge=0,
        le=1
    )

    uncertainty: float = Field(
        default=0.0,
        ge=0,
        le=1
    )

    visibility: float = Field(
        default=0.0,
        ge=0,
        le=1
    )

    urgency: float = Field(
        default=0.0,
        ge=0,
        le=1
    )


@app.post("/api/scheduler/priority")
def scheduler_priority(
    observation: PriorityInput
):

    return calculate_priority(

        conjunction=observation.conjunction,

        anomaly=observation.anomaly,

        uncertainty=observation.uncertainty,

        visibility=observation.visibility,

        urgency=observation.urgency
    )
