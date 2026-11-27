
from datetime import (
    datetime,
    timezone,
    timedelta
)

import pytest

from backend.scheduler import (

    calculate_priority,

    classify_priority,

    calculate_urgency,

    calculate_uncertainty_score,

    preferred_sensors,

    has_conflict,

    find_available_slot,

    allocate_observations,

    validate_schedule,

    collect_anomaly_risks,

    collect_conjunction_risks,

    generate_schedule
)


BASE_TIME = datetime(
    2026, 1, 1,
    0, 0, 0,
    tzinfo=timezone.utc
)


# ==========================================
# TEST FIXTURES
# ==========================================

def make_window(
    sensor="radar",
    start_seconds=0,
    end_seconds=300
):

    return {

        "sensor":
            sensor,

        "start_time": (
            BASE_TIME
            + timedelta(
                seconds=start_seconds
            )
        ).isoformat(),

        "end_time": (
            BASE_TIME
            + timedelta(
                seconds=end_seconds
            )
        ).isoformat(),

        "max_elevation_deg":
            45.0,

        "min_range_km":
            800.0
    }


def make_candidate(
    norad_id=25544,
    priority=90,
    windows=None,
    sensors=None
):

    if windows is None:

        windows = [
            make_window()
        ]

    if sensors is None:

        sensors = [
            "radar"
        ]

    return {

        "norad_id":
            norad_id,

        "satellite_name":
            f"SAT-{norad_id}",

        "priority_score":
            priority,

        "priority_level":
            classify_priority(priority),

        "conjunction_risk":
            "NORMAL",

        "anomaly_severity":
            "NORMAL",

        "anomaly_type":
            "NORMAL",

        "uncertainty_km":
            0.5,

        "preferred_sensors":
            sensors,

        "visibility_windows":
            windows,

        "tca":
            None,

        "explanation":
            "Automated test target"
    }


# ==========================================
# PRIORITY TESTS
# ==========================================

def test_maximum_priority():

    result = calculate_priority(
        1, 1, 1, 1, 1
    )

    assert (
        result["priority_score"]
        == 100
    )

    assert (
        result["priority_level"]
        == "CRITICAL"
    )


def test_minimum_priority():

    result = calculate_priority()

    assert (
        result["priority_score"]
        == 0
    )

    assert (
        result["priority_level"]
        == "LOW"
    )


def test_weighted_priority():

    result = calculate_priority(

        conjunction=1,

        anomaly=0.75,

        uncertainty=0.6,

        visibility=1,

        urgency=0.8
    )

    assert (
        result["priority_score"]
        == pytest.approx(83.75)
    )

    assert (
        sum(
            result[
                "contributions"
            ].values()
        )
        == pytest.approx(83.75)
    )


@pytest.mark.parametrize(

    "score, expected",

    [

        (0, "LOW"),

        (39.99, "LOW"),

        (40, "MEDIUM"),

        (60, "HIGH"),

        (80, "CRITICAL"),

        (100, "CRITICAL")
    ]
)
def test_priority_levels(
    score,
    expected
):

    assert (
        classify_priority(score)
        == expected
    )


def test_priority_clipping():

    result = calculate_priority(

        conjunction=5,

        anomaly=-1
    )

    assert (
        result["priority_score"]
        == 35
    )


# ==========================================
# URGENCY AND UNCERTAINTY
# ==========================================

def test_urgency():

    tca = (
        BASE_TIME
        + timedelta(hours=12)
    )

    assert calculate_urgency(

        tca,

        BASE_TIME

    ) == pytest.approx(0.5)


def test_uncertainty_score():

    assert (
        calculate_uncertainty_score(1)
        == pytest.approx(0.5)
    )

    assert (
        calculate_uncertainty_score(10)
        == 1
    )

    assert (
        calculate_uncertainty_score(None)
        == 0
    )


# ==========================================
# SENSOR PREFERENCE
# ==========================================

@pytest.mark.parametrize(

    "anomaly_type, expected",

    [

        ("LIDAR_ANOMALY", "radar"),

        ("RADAR_ANOMALY", "lidar"),

        ("OPTICAL_ANOMALY", "radar"),

        ("VELOCITY_ANOMALY", "radar"),

        ("ORBITAL_DEVIATION", "radar")
    ]
)
def test_sensor_preference(
    anomaly_type,
    expected
):

    result = preferred_sensors(
        anomaly_type=anomaly_type
    )

    assert result[0] == expected


# ==========================================
# CONFLICT DETECTION
# ==========================================

def test_sensor_conflict():

    schedule = [

        {
            "sensor": "radar",

            "start_time":
                BASE_TIME.isoformat(),

            "end_time": (
                BASE_TIME
                + timedelta(seconds=60)
            ).isoformat()
        }
    ]

    assert has_conflict(

        "radar",

        BASE_TIME
        + timedelta(seconds=30),

        BASE_TIME
        + timedelta(seconds=90),

        schedule
    )

    # Different sensors can operate
    # simultaneously.

    assert not has_conflict(

        "lidar",

        BASE_TIME
        + timedelta(seconds=30),

        BASE_TIME
        + timedelta(seconds=90),

        schedule
    )


def test_setup_time():

    schedule = [

        {
            "sensor": "radar",

            "start_time":
                BASE_TIME.isoformat(),

            "end_time": (
                BASE_TIME
                + timedelta(seconds=60)
            ).isoformat()
        }
    ]

    assert has_conflict(

        "radar",

        BASE_TIME
        + timedelta(seconds=65),

        BASE_TIME
        + timedelta(seconds=125),

        schedule
    )

    assert not has_conflict(

        "radar",

        BASE_TIME
        + timedelta(seconds=70),

        BASE_TIME
        + timedelta(seconds=130),

        schedule
    )


# ==========================================
# AVAILABLE SLOT
# ==========================================

def test_available_slot():

    window = make_window()

    slot = find_available_slot(
        window,
        []
    )

    assert slot is not None

    assert (
        slot["duration_seconds"]
        == 60
    )


def test_short_window():

    window = make_window(
        end_seconds=20
    )

    assert (
        find_available_slot(
            window,
            []
        )
        is None
    )


def test_observation_deadline():

    window = make_window()

    deadline = (
        BASE_TIME
        + timedelta(seconds=30)
    )

    assert (
        find_available_slot(
            window,
            [],
            deadline=deadline
        )
        is None
    )


# ==========================================
# PRIORITY-BASED ALLOCATION
# ==========================================

def test_high_priority_first():

    high = make_candidate(

        norad_id=1,

        priority=95
    )

    low = make_candidate(

        norad_id=2,

        priority=20
    )

    schedule, unscheduled = (
        allocate_observations(
            [low, high]
        )
    )

    assert len(schedule) == 2

    assert len(unscheduled) == 0

    assert (
        schedule[0]["norad_id"]
        == 1
    )

    assert validate_schedule(
        schedule
    )["valid"]


def test_no_visibility():

    candidate = make_candidate(
        windows=[]
    )

    schedule, unscheduled = (
        allocate_observations(
            [candidate]
        )
    )

    assert len(schedule) == 0

    assert len(unscheduled) == 1


def test_schedule_capacity():

    candidates = [

        make_candidate(
            norad_id=1
        ),

        make_candidate(
            norad_id=2
        )
    ]

    schedule, unscheduled = (
        allocate_observations(

            candidates,

            max_tasks=1
        )
    )

    assert len(schedule) == 1

    assert len(unscheduled) == 1


def test_multiple_sensors():

    radar_target = make_candidate(

        norad_id=1,

        windows=[
            make_window("radar")
        ],

        sensors=["radar"]
    )

    lidar_target = make_candidate(

        norad_id=2,

        windows=[
            make_window("lidar")
        ],

        sensors=["lidar"]
    )

    schedule, _ = (
        allocate_observations(
            [
                radar_target,
                lidar_target
            ]
        )
    )

    assert len(schedule) == 2

    assert (
        schedule[0]["start_time"]
        == schedule[1]["start_time"]
    )

    assert validate_schedule(
        schedule
    )["valid"]


def test_conflicting_targets():

    window = make_window(
        end_seconds=60
    )

    candidates = [

        make_candidate(

            norad_id=1,

            priority=95,

            windows=[window]
        ),

        make_candidate(

            norad_id=2,

            priority=20,

            windows=[window]
        )
    ]

    schedule, unscheduled = (
        allocate_observations(
            candidates
        )
    )

    assert len(schedule) == 1

    assert len(unscheduled) == 1

    assert (
        schedule[0]["norad_id"]
        == 1
    )


# ==========================================
# VALIDATION TESTS
# ==========================================

def test_invalid_schedule_detection():

    candidate = make_candidate()

    schedule, _ = (
        allocate_observations(
            [candidate]
        )
    )

    duplicate = schedule[0].copy()

    duplicate["task_id"] = "DUPLICATE"

    schedule.append(duplicate)

    validation = validate_schedule(
        schedule
    )

    assert not validation["valid"]

    assert (
        validation["total_violations"]
        >= 1
    )


def test_conjunction_deadline():

    candidate = make_candidate()

    candidate[
        "conjunction_risk"
    ] = "CRITICAL"

    candidate["tca"] = (

        BASE_TIME
        + timedelta(seconds=30)

    ).isoformat()

    schedule, unscheduled = (
        allocate_observations(
            [candidate]
        )
    )

    assert len(schedule) == 0

    assert len(unscheduled) == 1


# ==========================================
# ALERT INTEGRATION TESTS
# ==========================================

def test_conjunction_risk_collection():

    alerts = [

        {
            "object_a": {
                "norad_id": 25544
            },

            "object_b": {
                "norad_id": "SYN-DEBRIS-001"
            },

            "risk_level":
                "CRITICAL",

            "miss_distance_km":
                0.45,

            "tca": (
                BASE_TIME
                + timedelta(minutes=10)
            ).isoformat(),

            "simulated":
                True
        }
    ]

    result = collect_conjunction_risks(

        alerts,

        BASE_TIME
    )

    assert (
        result[25544]["risk_level"]
        == "CRITICAL"
    )

    assert len(result) == 1


def test_anomaly_risk_collection():

    alerts = [

        {
            "norad_id": 25544,

            "severity": "MEDIUM",

            "anomaly_score": 0.1
        },

        {
            "norad_id": 25544,

            "severity": "HIGH",

            "anomaly_score": 0.5
        }
    ]

    result = collect_anomaly_risks(
        alerts
    )

    assert (
        result[25544]["severity"]
        == "HIGH"
    )


def test_future_anomaly_excluded():

    alerts = [

        {
            "norad_id": 25544,

            "severity": "CRITICAL",

            "timestamp": (
                BASE_TIME
                + timedelta(minutes=10)
            ).isoformat()
        }
    ]

    result = collect_anomaly_risks(

        alerts,

        start_time=BASE_TIME
    )

    assert result == {}


# ==========================================
# FULL SCHEDULER INTEGRATION
# ==========================================

def test_scheduler_integration():

    result = generate_schedule(

        hours=1,

        max_objects=3,

        max_tasks=3,

        mode="demo",

        include_fusion=False
    )

    assert result["simulated"] is True

    assert (
        result["total_candidates"]
        <= 3
    )

    assert (

        result["total_scheduled"]

        + result["total_unscheduled"]

        == result["total_candidates"]
    )

    assert (
        result["validation"]["valid"]
        is True
    )

    for candidate in result[
        "candidates"
    ]:

        assert (
            0
            <= candidate["priority_score"]
            <= 100
        )


# ==========================================
# API TEST
# ==========================================

def test_scheduler_api():

    from fastapi.testclient import (
        TestClient
    )

    from backend.main import app

    client = TestClient(app)

    response = client.get(
        "/api/scheduler/config"
    )

    assert response.status_code == 200

    assert (
        "priority_weights"
        in response.json()
    )

    response = client.post(

        "/api/scheduler/priority",

        json={

            "conjunction": 1,

            "anomaly": 1,

            "uncertainty": 1,

            "visibility": 1,

            "urgency": 1
        }
    )

    assert response.status_code == 200

    assert (
        response.json()[
            "priority_score"
        ]
        == 100
    )

    invalid = client.post(

        "/api/scheduler/priority",

        json={
            "conjunction": 2
        }
    )

    assert invalid.status_code == 422
