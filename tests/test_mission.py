
import pytest

from backend.cache import TTLCache

from backend.mission import (

    build_alert_feed,

    normalise_conjunction_alert,

    normalise_anomaly_alert,

    safe_execute,

    get_mission_overview
)


# ==========================================
# ALERT FIXTURES
# ==========================================

@pytest.fixture
def conjunction_alert():

    return {

        "alert_id":
            "CON-001",

        "risk_level":
            "CRITICAL",

        "object_a": {
            "norad_id": 25544,
            "name": "ISS"
        },

        "object_b": {
            "norad_id": 12345,
            "name": "DEBRIS"
        },

        "tca":
            "2026-01-01T12:00:00+00:00",

        "miss_distance_km":
            0.45,

        "relative_velocity_km_s":
            12.0,

        "explanation":
            "Close approach predicted",

        "recommended_action":
            "Additional observation",

        "simulated":
            True
    }


@pytest.fixture
def anomaly_alert():

    return {

        "alert_id":
            "ANO-001",

        "norad_id":
            25544,

        "satellite_name":
            "ISS",

        "severity":
            "HIGH",

        "suspected_type":
            "LIDAR_ANOMALY",

        "anomaly_score":
            0.8,

        "timestamp":
            "2026-01-01T10:00:00+00:00",

        "detection_source":
            "STATISTICAL_RESIDUAL",

        "explanation":
            "Large LiDAR residual",

        "recommended_action":
            "Cross-check using radar",

        "simulated":
            True
    }


# ==========================================
# CACHE TESTS
# ==========================================

def test_cache_hit():

    cache = TTLCache(
        ttl_seconds=60
    )

    calls = []

    def factory():

        calls.append(1)

        return {"value": 42}

    first = cache.get_or_create(
        "key",
        factory
    )

    second = cache.get_or_create(
        "key",
        factory
    )

    assert first == second

    assert len(calls) == 1

    assert cache.stats()["hits"] == 1

    assert cache.stats()["misses"] == 1


def test_cache_clear():

    cache = TTLCache()

    cache.get_or_create(
        "key",
        lambda: 42
    )

    cache.clear()

    assert cache.stats()["entries"] == 0


def test_cache_expiry(monkeypatch):

    import backend.cache as cache_module

    clock = [0.0]

    monkeypatch.setattr(

        cache_module,

        "monotonic",

        lambda: clock[0]
    )

    cache = TTLCache(
        ttl_seconds=5
    )

    first = cache.get_or_create(
        "key",
        lambda: 1
    )

    clock[0] = 6.0

    second = cache.get_or_create(
        "key",
        lambda: 2
    )

    assert first == 1

    assert second == 2


def test_cache_maxsize():

    cache = TTLCache(
        maxsize=2
    )

    for index in range(3):

        cache.get_or_create(

            index,

            lambda: index
        )

    assert cache.stats()["entries"] == 2


def test_cache_mutation_protection():

    cache = TTLCache()

    result = cache.get_or_create(

        "key",

        lambda: {"items": [1, 2]}
    )

    result["items"].append(3)

    cached = cache.get_or_create(

        "key",

        lambda: {}
    )

    assert cached["items"] == [1, 2]


def test_cache_failed_factory():

    cache = TTLCache()

    def fail():

        raise RuntimeError(
            "Simulated failure"
        )

    with pytest.raises(RuntimeError):

        cache.get_or_create(
            "key",
            fail
        )

    assert cache.stats()["entries"] == 0


# ==========================================
# ALERT NORMALISATION
# ==========================================

def test_conjunction_normalisation(
    conjunction_alert
):

    result = normalise_conjunction_alert(
        conjunction_alert
    )

    assert result["category"] == "CONJUNCTION"

    assert result["severity"] == "CRITICAL"

    assert result["priority"] == 4

    assert result["simulated"] is True

    assert (
        result["details"]["miss_distance_km"]
        == 0.45
    )


def test_anomaly_normalisation(
    anomaly_alert
):

    result = normalise_anomaly_alert(
        anomaly_alert
    )

    assert result["category"] == "ANOMALY"

    assert result["severity"] == "HIGH"

    assert result["priority"] == 3

    assert result["simulated"] is True


def test_alert_priority_order(

    conjunction_alert,

    anomaly_alert
):

    result = build_alert_feed(

        [conjunction_alert],

        [anomaly_alert]
    )

    assert result["total_alerts"] == 2

    assert (
        result["alerts"][0]["severity"]
        == "CRITICAL"
    )


def test_severity_filter(

    conjunction_alert,

    anomaly_alert
):

    result = build_alert_feed(

        [conjunction_alert],

        [anomaly_alert],

        severity="HIGH"
    )

    assert result["total_alerts"] == 1

    assert (
        result["alerts"][0]["category"]
        == "ANOMALY"
    )


def test_category_filter(

    conjunction_alert,

    anomaly_alert
):

    result = build_alert_feed(

        [conjunction_alert],

        [anomaly_alert],

        category="CONJUNCTION"
    )

    assert result["total_alerts"] == 1


def test_alert_deduplication(
    conjunction_alert
):

    result = build_alert_feed(

        [
            conjunction_alert,
            conjunction_alert
        ],

        []
    )

    assert result["total_alerts"] == 1


def test_alert_limit(

    conjunction_alert,

    anomaly_alert
):

    result = build_alert_feed(

        [conjunction_alert],

        [anomaly_alert],

        limit=1
    )

    assert result["total_alerts"] == 2

    assert result["returned_alerts"] == 1

    assert len(result["alerts"]) == 1


def test_empty_alert_feed():

    result = build_alert_feed()

    assert result["total_alerts"] == 0

    assert result["alerts"] == []


# ==========================================
# GRACEFUL DEGRADATION
# ==========================================

def test_safe_execute_success():

    status = {}

    warnings = []

    result = safe_execute(

        "test_module",

        lambda: {"value": 1},

        status,

        warnings
    )

    assert result["value"] == 1

    assert (
        status["test_module"]["status"]
        == "AVAILABLE"
    )

    assert warnings == []


def test_safe_execute_failure():

    status = {}

    warnings = []

    def fail():

        raise RuntimeError(
            "Simulated module failure"
        )

    result = safe_execute(

        "test_module",

        fail,

        status,

        warnings
    )

    assert result is None

    assert (
        status["test_module"]["status"]
        == "DEGRADED"
    )

    assert len(warnings) == 1


# ==========================================
# FULL INTEGRATION
# ==========================================

def test_mission_integration():

    result = get_mission_overview(

        hours=1,

        max_objects=3,

        mode="demo",

        include_fusion=False
    )

    assert (
        result["application"]
        == "CosmoTrack"
    )

    assert result["catalogue"] is not None

    assert result["selected_satellite"] is not None

    assert result["conjunctions"] is not None

    assert result["anomalies"] is not None

    assert result["scheduler"] is not None

    assert result["alert_feed"] is not None

    assert (
        result["module_status"]["fusion"]["status"]
        == "DISABLED"
    )

    assert (
        result["scheduler"]["validation"]["valid"]
        is True
    )


def test_catalogue_mode():

    result = get_mission_overview(

        hours=1,

        max_objects=2,

        mode="catalogue",

        include_fusion=False
    )

    assert result["anomalies"] is None

    assert (
        result["alert_feed"]["anomaly_alerts"]
        == 0
    )

    assert (
        result["data_provenance"][
            "synthetic_scenarios_enabled"
        ]
        is False
    )


# ==========================================
# FASTAPI FIXTURE
# ==========================================

@pytest.fixture
def client():

    from fastapi.testclient import TestClient

    from backend.main import app

    return TestClient(app)


# ==========================================
# API TESTS
# ==========================================

def test_system_info_api(client):

    response = client.get(
        "/api/system/info"
    )

    assert response.status_code == 200

    assert (
        response.json()["application"]
        == "CosmoTrack"
    )


def test_cache_api(client):

    response = client.get(
        "/api/system/cache"
    )

    assert response.status_code == 200

    assert "cache" in response.json()


def test_readiness_api(client):

    response = client.get(
        "/api/system/ready"
    )

    assert response.status_code == 200

    assert response.json()["ready"] is True


def test_mission_api(client):

    response = client.get(

        "/api/mission/overview",

        params={

            "hours": 1,

            "max_objects": 3,

            "include_fusion": False
        }
    )

    assert response.status_code == 200

    assert "alert_feed" in response.json()

    assert "X-Process-Time-Ms" in response.headers


def test_invalid_mission_parameters(client):

    response = client.get(

        "/api/mission/overview",

        params={"hours": 100}
    )

    assert response.status_code == 422


def test_invalid_mode(client):

    response = client.get(

        "/api/mission/overview",

        params={"mode": "invalid"}
    )

    assert response.status_code == 422


def test_alert_api(client):

    response = client.get(

        "/api/mission/alerts",

        params={

            "hours": 1,

            "limit": 5,

            "category": "ANOMALY"
        }
    )

    assert response.status_code == 200

    result = response.json()

    assert len(result["alerts"]) <= 5

    for alert in result["alerts"]:

        assert alert["category"] == "ANOMALY"


def test_cors(client):

    response = client.options(

        "/api/mission/overview",

        headers={

            "Origin":
                "http://localhost:5173",

            "Access-Control-Request-Method":
                "GET"
        }
    )

    assert response.status_code == 200

    assert (

        response.headers.get(
            "access-control-allow-origin"
        )

        == "http://localhost:5173"
    )


def test_openapi_documentation(client):

    response = client.get(
        "/openapi.json"
    )

    assert response.status_code == 200

    paths = response.json()["paths"]

    assert "/api/mission/overview" in paths

    assert "/api/mission/alerts" in paths

