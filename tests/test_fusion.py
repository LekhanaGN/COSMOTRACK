
import numpy as np

from backend.fusion import run_fusion
from backend.orbit import get_satellites
from backend.sensors import find_next_pass


def test_ekf_fusion():

    satellites = get_satellites()

    norad_id = (
        25544 if 25544 in satellites
        else next(iter(satellites))
    )

    result = run_fusion(
        norad_id,
        duration_seconds=120,
        step_seconds=10
    )

    assert result["total_samples"] > 0

    assert result["measurement_updates"] > 0

    for reading in result["results"]:

        position = np.array(
            reading["fused_position_km"]
        )

        covariance = np.array(
            reading["covariance"]
        )

        assert np.isfinite(position).all()

        assert covariance.shape == (6, 6)

        assert np.isfinite(covariance).all()

        assert np.allclose(
            covariance,
            covariance.T,
            atol=1e-8
        )

        eigenvalues = np.linalg.eigvalsh(
            covariance
        )

        assert np.min(eigenvalues) >= -1e-8

        assert (
            reading["position_uncertainty_km"]
            >= 0
        )


def test_sensor_dropout():

    norad_id = next(
        iter(get_satellites())
    )

    start_time = find_next_pass(norad_id)

    for sensor in [
        "lidar",
        "radar",
        "optical"
    ]:

        result = run_fusion(
            norad_id,
            duration_seconds=60,
            dropout_sensor=sensor,
            start_time=start_time
        )

        assert result["total_samples"] > 0

        assert result["measurement_updates"] > 0

        assert np.isfinite(
            result["metrics"]["fused_rmse_km"]
        )
