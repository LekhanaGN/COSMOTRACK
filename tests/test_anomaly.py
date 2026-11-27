
import numpy as np
import pytest

from backend.anomaly import (
    FEATURE_NAMES,
    NORMAL_SIGMA,
    ANOMALY_TYPES,
    generate_normal_data,
    generate_evaluation_data,
    get_detector,
    analyze_observation,
    explain_anomaly,
    classify_severity,
    extract_residual_features,
    evaluate_model,
    run_anomaly_demo
)


# ==========================================
# TRAINING DATA TEST
# ==========================================

def test_training_data():

    data = generate_normal_data(
        samples=100
    )

    assert data.shape == (100, 6)

    assert np.isfinite(data).all()

    assert np.all(
        data[:, [0, 1, 5]] >= 0
    )


# ==========================================
# MODEL TRAINING TEST
# ==========================================

def test_model_training():

    detector = get_detector()

    assert detector.trained is True

    assert (
        detector.model.n_estimators
        == 200
    )


# ==========================================
# NORMAL OBSERVATION TEST
# ==========================================

def test_normal_observation():

    observation = [
        120,
        0.8,
        5,
        15,
        0.3,
        0.01
    ]

    result = analyze_observation(
        observation
    )

    assert result["is_anomaly"] is False

    assert result["severity"] == "NORMAL"

    assert result["anomaly_score"] <= 0


# ==========================================
# STRONG ANOMALY TEST
# ==========================================

def test_strong_anomaly():

    observation = [
        4500,
        35,
        100,
        500,
        15,
        0.3
    ]

    result = analyze_observation(
        observation
    )

    assert result["is_anomaly"] is True

    assert result["severity"] == "CRITICAL"

    assert result["anomaly_score"] > 0


# ==========================================
# EXPLANATION TEST
# ==========================================

def test_explanation():

    observation = [
        120,
        0.8,
        150,
        15,
        0.3,
        0.01
    ]

    result = explain_anomaly(
        observation
    )

    assert (
        result["suspected_type"]
        == "LIDAR_ANOMALY"
    )

    assert (
        result["dominant_feature"]
        == "lidar_residual_m"
    )

    assert (
        result["max_sigma_multiple"]
        == pytest.approx(15.0)
    )


# ==========================================
# SEVERITY CLASSIFICATION TEST
# ==========================================

@pytest.mark.parametrize(
    "is_anomaly, sigma, expected",
    [
        (False, 15, "NORMAL"),
        (True, 2, "LOW"),
        (True, 3, "MEDIUM"),
        (True, 6, "HIGH"),
        (True, 10, "CRITICAL")
    ]
)
def test_severity(
    is_anomaly,
    sigma,
    expected
):

    assert classify_severity(
        is_anomaly,
        sigma
    ) == expected


# ==========================================
# EVALUATION DATA TEST
# ==========================================

def test_evaluation_data():

    data, labels, types = (
        generate_evaluation_data(
            samples=500,
            anomaly_fraction=0.10
        )
    )

    assert data.shape == (500, 6)

    assert len(labels) == 500

    assert labels.sum() == 50

    assert set(ANOMALY_TYPES).issubset(
        set(types)
    )


# ==========================================
# MODEL PERFORMANCE TEST
# ==========================================

def test_model_performance():

    metrics = evaluate_model()

    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate"
    ]:

        assert 0 <= metrics[key] <= 1

    cm = metrics["confusion_matrix"]

    assert sum(cm.values()) == 500

    assert (
        cm["true_positive"]
        + cm["false_negative"]
        == 50
    )

    assert (
        cm["true_positive"]
        + cm["false_positive"]
        == metrics["detected_anomalies"]
    )

    # Basic controlled-simulation
    # performance requirement.

    assert metrics["recall"] >= 0.50

    assert metrics["f1_score"] >= 0.50


# ==========================================
# RESIDUAL EXTRACTION TEST
# ==========================================

def test_residual_extraction():

    features = extract_residual_features(

        expected_position_km=[
            7000, 0, 0
        ],

        observed_position_km=[
            7000.3, 0, 0
        ],

        expected_velocity_km_s=[
            0, 7.5, 0
        ],

        observed_velocity_km_s=[
            0, 7.502, 0
        ],

        expected_range_km=1000,

        lidar_range_km=1000.01,

        radar_range_km=1000.05,

        expected_range_rate_km_s=1,

        radar_range_rate_km_s=1.001,

        expected_azimuth_deg=359,

        observed_azimuth_deg=1,

        expected_elevation_deg=0,

        observed_elevation_deg=0
    )

    expected = np.array([
        300,
        2,
        10,
        50,
        1,
        2
    ])

    np.testing.assert_allclose(
        features,
        expected,
        atol=1e-6
    )


# ==========================================
# INVALID INPUT TEST
# ==========================================

def test_invalid_input():

    with pytest.raises(ValueError):

        analyze_observation(
            [1, 2, 3]
        )

    with pytest.raises(ValueError):

        analyze_observation(
            [1, 2, 3, 4, 5, np.nan]
        )


# ==========================================
# SATELLITE DEMO TEST
# ==========================================

def test_satellite_demo():

    result = run_anomaly_demo(
        samples=60
    )

    assert result["simulated"] is True

    assert len(
        result["timeline"]
    ) == 60

    assert (
        result["injected_anomalies"]
        == 15
    )

    assert (
        result["detected_anomalies"]
        == len(result["alerts"])
    )

    assert sum(
        result["severity_counts"].values()
    ) == result["detected_anomalies"]

    for alert in result["alerts"]:

        assert alert["is_anomaly"] is True

        assert alert["simulated"] is True

        assert len(
            alert["position_km"]
        ) == 3

        assert alert["explanation"]

        assert alert["recommended_action"]


# ==========================================
# REPRODUCIBILITY TEST
# ==========================================

def test_reproducibility():

    first = evaluate_model()

    second = evaluate_model()

    assert first == second


# ==========================================
# API TEST
# ==========================================

def test_anomaly_api():

    from fastapi.testclient import TestClient

    from backend.main import app

    client = TestClient(app)

    response = client.get(
        "/api/anomalies/metrics"
    )

    assert response.status_code == 200

    assert "f1_score" in response.json()

    payload = {
        "position_residual_m": 120,
        "velocity_residual_m_s": 0.8,
        "lidar_residual_m": 5,
        "radar_residual_m": 15,
        "radar_range_rate_residual_m_s": 0.3,
        "optical_residual_deg": 0.01
    }

    response = client.post(
        "/api/anomalies/analyze",
        json=payload
    )

    assert response.status_code == 200

    assert "is_anomaly" in response.json()
@pytest.mark.parametrize("feature_index", range(6))
def test_single_feature_anomaly(feature_index):

    observation = NORMAL_SIGMA * 0.5

    observation[feature_index] = (
        NORMAL_SIGMA[feature_index] * 8
    )

    result = analyze_observation(observation)

    assert result["is_anomaly"] is True
    assert result["severity"] == "HIGH"

def test_hybrid_performance():

    from backend.anomaly import evaluate_model

    metrics = evaluate_model()

    ml_metrics = metrics["ml_only_metrics"]

    assert metrics["recall"] >= ml_metrics["recall"]

    assert (
        metrics["confusion_matrix"]["false_negative"]
        <= ml_metrics["confusion_matrix"]["false_negative"]
    )


def test_detection_source():

    from backend.anomaly import analyze_observation

    observation = [
        120.0,
        0.8,
        150.0,
        15.0,
        0.3,
        0.01
    ]

    result = analyze_observation(observation)

    assert result["is_anomaly"] is True

    assert result["statistical_detected"] is True

    assert result["detection_source"] in [
        "STATISTICAL_RESIDUAL",
        "ML_AND_STATISTICAL"
    ]


def test_negative_magnitude():

    import pytest

    from backend.anomaly import analyze_observation

    with pytest.raises(ValueError):

        analyze_observation([
            -100.0,
            0.8,
            5.0,
            15.0,
            0.3,
            0.01
        ])
