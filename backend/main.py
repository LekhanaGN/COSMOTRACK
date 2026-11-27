
from fastapi import FastAPI, HTTPException, Query

from backend.orbit import (
    list_satellites,
    get_state,
    get_trajectory
)

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
