
from datetime import datetime, timezone

import numpy as np
import pytest

from backend.collision import (
    classify_risk,
    refine_closest_approach,
    synthetic_encounter,
    screen_conjunctions
)


# ==========================================
# RISK CLASSIFICATION TESTS
# ==========================================

@pytest.mark.parametrize(
    "distance, expected",
    [
        (0.0, "CRITICAL"),
        (0.45, "CRITICAL"),
        (0.99, "CRITICAL"),
        (1.0, "HIGH"),
        (2.5, "HIGH"),
        (5.0, "MEDIUM"),
        (7.5, "MEDIUM"),
        (10.0, "LOW"),
        (25.0, "LOW")
    ]
)
def test_risk_classification(
    distance,
    expected
):

    assert (
        classify_risk(distance)
        == expected
    )


# ==========================================
# INVALID INPUT TEST
# ==========================================

def test_invalid_distance():

    with pytest.raises(ValueError):

        classify_risk(-1.0)

    with pytest.raises(ValueError):

        classify_risk(float("nan"))


# ==========================================
# KNOWN CLOSEST APPROACH TEST
# ==========================================

def test_closest_approach():

    # Object A remains at the origin.
    def position_a(t):

        return np.array([
            0.0, 0.0, 0.0
        ])

    # Object B passes with a known
    # minimum separation of 0.5 km
    # at exactly 100 seconds.
    def position_b(t):

        return np.array([
            10.0 * (t - 100.0),
            0.5,
            0.0
        ])

    tca, distance = refine_closest_approach(
        position_a,
        position_b,
        0,
        200
    )

    assert abs(tca - 100.0) < 0.01

    assert abs(distance - 0.5) < 0.001


# ==========================================
# SCREENING WINDOW BOUNDARY TEST
# ==========================================

def test_boundary_minimum():

    def position_a(t):

        return np.zeros(3)

    def position_b(t):

        return np.array([
            t + 2.0,
            0.0,
            0.0
        ])

    tca, distance = refine_closest_approach(
        position_a,
        position_b,
        0,
        100
    )

    assert abs(tca) < 0.01

    assert abs(distance - 2.0) < 0.001


# ==========================================
# SYNTHETIC ENCOUNTER TEST
# ==========================================

def test_synthetic_encounter():

    result = synthetic_encounter(
        miss_distance_km=0.45,
        relative_speed_km_s=12.0,
        tca_seconds=600
    )

    alert = result["alert"]

    assert result["simulated"] is True

    assert alert["simulated"] is True

    assert alert["risk_level"] == "CRITICAL"

    assert abs(
        alert["miss_distance_km"] - 0.45
    ) < 0.001

    assert abs(
        alert["seconds_to_tca"] - 600
    ) < 0.01

    assert abs(
        alert["relative_velocity_km_s"] - 12
    ) < 0.001

    assert (
        alert["collision_probability"]
        is None
    )

    assert len(result["trajectory"]) > 0

    for point in result["trajectory"]:

        assert point["separation_km"] >= 0

        assert np.isfinite(
            point["separation_km"]
        )


# ==========================================
# REAL CATALOGUE SCREENING TEST
# ==========================================

def test_catalogue_screening():

    result = screen_conjunctions(
        hours=0.25,
        step_seconds=60,
        screening_distance_km=25.0
    )

    assert result["total_objects"] > 0

    assert result["valid_objects"] >= 2

    assert result["screened_pairs"] > 0

    assert result["total_alerts"] >= 0

    assert sum(
        result["risk_counts"].values()
    ) == result["total_alerts"]

    assert (
        result["collision_probability_available"]
        is False
    )

    for alert in result["alerts"]:

        assert (
            0 <= alert["miss_distance_km"] <= 25
        )

        assert (
            0 <= alert["seconds_to_tca"] <= 900
        )

        assert np.isfinite(
            alert["relative_velocity_km_s"]
        )

        assert (
            alert["object_a"]["id"]
            != alert["object_b"]["id"]
        )

        assert alert["simulated"] is False

        assert (
            alert["risk_level"]
            == classify_risk(
                alert["miss_distance_km"]
            )
        )


# ==========================================
# PRIORITY CHANGE TEST
# ==========================================

def test_priority_change():

    start = datetime.now(timezone.utc)

    critical = synthetic_encounter(
        miss_distance_km=0.45,
        start_time=start
    )

    low = synthetic_encounter(
        miss_distance_km=15.0,
        start_time=start
    )

    assert (
        critical["alert"]["priority"]
        >
        low["alert"]["priority"]
    )
