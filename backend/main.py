
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
