
from datetime import datetime, timezone, timedelta
from functools import lru_cache

import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)

from backend.orbit import get_satellites, get_state


# ==========================================
# CONFIGURATION
# ==========================================

RANDOM_SEED = 42

MODEL_NAME = (
    "Hybrid Isolation Forest + "
    "Statistical Residual Detection"
)

TRAINING_SAMPLES = 2000

STATISTICAL_THRESHOLD = 5.0

FEATURE_NAMES = [
    "position_residual_m",
    "velocity_residual_m_s",
    "lidar_residual_m",
    "radar_residual_m",
    "radar_range_rate_residual_m_s",
    "optical_residual_deg"
]

FEATURE_LABELS = [
    "Position residual",
    "Velocity residual",
    "LiDAR range residual",
    "Radar range residual",
    "Radar range-rate residual",
    "Optical angular residual"
]

FEATURE_UNITS = [
    "m",
    "m/s",
    "m",
    "m",
    "m/s",
    "deg"
]

NORMAL_SIGMA = np.array([
    300.0,
    2.0,
    10.0,
    50.0,
    1.0,
    0.02
])

ANOMALY_TYPES = [
    "ORBITAL_DEVIATION",
    "VELOCITY_ANOMALY",
    "LIDAR_ANOMALY",
    "RADAR_ANOMALY",
    "OPTICAL_ANOMALY"
]


# ==========================================
# NORMAL TRAINING DATA
# ==========================================

def generate_normal_data(
    samples=2000,
    seed=RANDOM_SEED
):

    if samples < 1:
        raise ValueError(
            "Samples must be positive"
        )

    rng = np.random.default_rng(seed)

    data = rng.normal(
        loc=0.0,
        scale=NORMAL_SIGMA,
        size=(samples, len(FEATURE_NAMES))
    )

    # Position, velocity and optical residuals
    # are represented as positive magnitudes.

    data[:, [0, 1, 5]] = np.abs(
        data[:, [0, 1, 5]]
    )

    return data


# ==========================================
# INJECT CONTROLLED ANOMALIES
# ==========================================

def inject_anomaly(
    observation,
    anomaly_type,
    rng,
    strength=8.0
):

    if anomaly_type not in ANOMALY_TYPES:
        raise ValueError(
            "Unknown anomaly type"
        )

    if not np.isfinite(strength) or strength <= 0:
        raise ValueError(
            "Strength must be positive and finite"
        )

    result = np.asarray(
        observation,
        dtype=float
    ).copy()

    magnitude = strength * rng.uniform(
        0.8, 1.2
    )

    if anomaly_type == "ORBITAL_DEVIATION":

        result[0] += (
            magnitude * NORMAL_SIGMA[0]
        )

        result[1] += (
            magnitude * NORMAL_SIGMA[1] * 0.5
        )

    elif anomaly_type == "VELOCITY_ANOMALY":

        result[1] += (
            magnitude * NORMAL_SIGMA[1]
        )

    elif anomaly_type == "LIDAR_ANOMALY":

        result[2] += (
            magnitude
            * NORMAL_SIGMA[2]
            * rng.choice([-1, 1])
        )

    elif anomaly_type == "RADAR_ANOMALY":

        result[3] += (
            magnitude
            * NORMAL_SIGMA[3]
            * rng.choice([-1, 1])
        )

        result[4] += (
            magnitude
            * NORMAL_SIGMA[4]
            * rng.choice([-1, 1])
        )

    elif anomaly_type == "OPTICAL_ANOMALY":

        result[5] += (
            magnitude * NORMAL_SIGMA[5]
        )

    return result


# ==========================================
# INDEPENDENT EVALUATION DATASET
# ==========================================

def generate_evaluation_data(
    samples=500,
    anomaly_fraction=0.10,
    strength=8.0,
    seed=123
):

    if samples < 2:
        raise ValueError(
            "At least two samples required"
        )

    if not 0 < anomaly_fraction < 1:
        raise ValueError(
            "Invalid anomaly fraction"
        )

    if not np.isfinite(strength) or strength <= 0:
        raise ValueError(
            "Strength must be positive and finite"
        )

    rng = np.random.default_rng(seed)

    data = generate_normal_data(
        samples=samples,
        seed=seed + 1
    )

    labels = np.zeros(
        samples,
        dtype=int
    )

    types = ["NORMAL"] * samples

    count = max(
        1,
        min(
            samples - 1,
            int(samples * anomaly_fraction)
        )
    )

    indices = rng.choice(
        samples,
        size=count,
        replace=False
    )

    for i, index in enumerate(indices):

        anomaly_type = ANOMALY_TYPES[
            i % len(ANOMALY_TYPES)
        ]

        data[index] = inject_anomaly(
            data[index],
            anomaly_type,
            rng,
            strength
        )

        labels[index] = 1

        types[index] = anomaly_type

    return data, labels, types


# ==========================================
# HYBRID ANOMALY DETECTOR
# ==========================================

class AnomalyDetector:

    def __init__(self):

        self.scaler = StandardScaler()

        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.02,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )

        self.trained = False

    # --------------------------------------
    # Train Isolation Forest
    # --------------------------------------

    def fit(self, training_data):

        training_data = np.asarray(
            training_data,
            dtype=float
        )

        if (
            training_data.ndim != 2
            or training_data.shape[1]
            != len(FEATURE_NAMES)
            or training_data.shape[0] < 2
            or not np.isfinite(
                training_data
            ).all()
        ):
            raise ValueError(
                "Invalid training data"
            )

        scaled = self.scaler.fit_transform(
            training_data
        )

        self.model.fit(scaled)

        self.trained = True

        return self

    # --------------------------------------
    # Validate observations
    # --------------------------------------

    def validate_observations(
        self,
        observations
    ):

        if not self.trained:
            raise RuntimeError(
                "Model is not trained"
            )

        observations = np.asarray(
            observations,
            dtype=float
        )

        observations = np.atleast_2d(
            observations
        )

        if (
            observations.ndim != 2
            or observations.shape[1]
            != len(FEATURE_NAMES)
            or observations.shape[0] == 0
            or not np.isfinite(
                observations
            ).all()
        ):
            raise ValueError(
                "Invalid observations"
            )

        if np.any(
            observations[:, [0, 1, 5]] < 0
        ):
            raise ValueError(
                "Position, velocity and optical "
                "magnitudes cannot be negative"
            )

        return observations

    # --------------------------------------
    # Detailed hybrid prediction
    # --------------------------------------

    def predict_details(
        self,
        observations
    ):

        observations = (
            self.validate_observations(
                observations
            )
        )

        scaled = self.scaler.transform(
            observations
        )

        # Layer 1:
        # Isolation Forest ML detection.
        #
        # Positive score = anomaly.

        ml_scores = (
            -self.model.decision_function(
                scaled
            )
        )

        ml_predictions = (
            ml_scores > 0
        ).astype(int)

        # Layer 2:
        # Statistical residual detection.
        #
        # Detect unusually large individual
        # sensor or orbital residuals.

        deviations = (
            np.abs(observations)
            / NORMAL_SIGMA
        )

        max_deviations = np.max(
            deviations,
            axis=1
        )

        statistical_scores = (
            max_deviations
            / STATISTICAL_THRESHOLD
        ) - 1.0

        statistical_predictions = (
            statistical_scores > 0
        ).astype(int)

        # Layer 3:
        # Combine both detection methods.
        #
        # Either detector can trigger an alert.

        combined_scores = np.maximum(
            ml_scores,
            statistical_scores
        )

        predictions = (
            combined_scores > 0
        ).astype(int)

        return {
            "predictions": predictions,

            "scores": combined_scores,

            "ml_predictions":
                ml_predictions,

            "ml_scores":
                ml_scores,

            "statistical_predictions":
                statistical_predictions,

            "statistical_scores":
                statistical_scores,

            "max_deviations":
                max_deviations
        }

    # --------------------------------------
    # Backward-compatible prediction method
    # --------------------------------------

    def predict(self, observations):

        details = self.predict_details(
            observations
        )

        return (
            details["predictions"],
            details["scores"]
        )


# ==========================================
# TRAIN MODEL ONCE AND CACHE IT
# ==========================================

@lru_cache(maxsize=1)
def get_detector():

    training_data = generate_normal_data(
        samples=TRAINING_SAMPLES,
        seed=RANDOM_SEED
    )

    detector = AnomalyDetector()

    detector.fit(training_data)

    return detector


# ==========================================
# RESIDUAL FEATURE EXTRACTION
# ==========================================

def extract_residual_features(
    expected_position_km,
    observed_position_km,
    expected_velocity_km_s,
    observed_velocity_km_s,
    expected_range_km,
    lidar_range_km,
    radar_range_km,
    expected_range_rate_km_s,
    radar_range_rate_km_s,
    expected_azimuth_deg,
    observed_azimuth_deg,
    expected_elevation_deg,
    observed_elevation_deg
):

    expected_position = np.asarray(
        expected_position_km,
        dtype=float
    )

    observed_position = np.asarray(
        observed_position_km,
        dtype=float
    )

    expected_velocity = np.asarray(
        expected_velocity_km_s,
        dtype=float
    )

    observed_velocity = np.asarray(
        observed_velocity_km_s,
        dtype=float
    )

    for vector in [
        expected_position,
        observed_position,
        expected_velocity,
        observed_velocity
    ]:

        if vector.shape != (3,):
            raise ValueError(
                "State vectors must contain "
                "three components"
            )

    # Position residual: km to metres.

    position_residual = (
        np.linalg.norm(
            observed_position
            - expected_position
        ) * 1000
    )

    # Velocity residual: km/s to m/s.

    velocity_residual = (
        np.linalg.norm(
            observed_velocity
            - expected_velocity
        ) * 1000
    )

    # LiDAR range residual.

    lidar_residual = (
        lidar_range_km
        - expected_range_km
    ) * 1000

    # Radar range residual.

    radar_residual = (
        radar_range_km
        - expected_range_km
    ) * 1000

    # Radar range-rate residual.

    range_rate_residual = (
        radar_range_rate_km_s
        - expected_range_rate_km_s
    ) * 1000

    # Handle azimuth wraparound.
    # Example: 359 degrees to 1 degree
    # corresponds to a 2-degree difference.

    azimuth_error = (
        (
            observed_azimuth_deg
            - expected_azimuth_deg
            + 180
        ) % 360
    ) - 180

    elevation_error = (
        observed_elevation_deg
        - expected_elevation_deg
    )

    # Local small-angle approximation.

    optical_residual = np.hypot(
        azimuth_error * np.cos(
            np.radians(
                expected_elevation_deg
            )
        ),
        elevation_error
    )

    features = np.array([
        position_residual,
        velocity_residual,
        lidar_residual,
        radar_residual,
        range_rate_residual,
        optical_residual
    ])

    if not np.isfinite(features).all():
        raise ValueError(
            "Non-finite residual values"
        )

    return features


# ==========================================
# IDENTIFY DETECTION SOURCE
# ==========================================

def get_detection_source(
    ml_detected,
    statistical_detected
):

    if ml_detected and statistical_detected:

        return "ML_AND_STATISTICAL"

    if ml_detected:

        return "ISOLATION_FOREST"

    if statistical_detected:

        return "STATISTICAL_RESIDUAL"

    return "NONE"


# ==========================================
# EXPLAINABLE ANOMALY ANALYSIS
# ==========================================

def explain_anomaly(observation):

    observation = np.asarray(
        observation,
        dtype=float
    )

    if (
        observation.shape
        != (len(FEATURE_NAMES),)
        or not np.isfinite(
            observation
        ).all()
    ):
        raise ValueError(
            "Invalid observation"
        )

    deviations = (
        np.abs(observation)
        / NORMAL_SIGMA
    )

    ranked_indices = np.argsort(
        deviations
    )[::-1]

    dominant = int(
        ranked_indices[0]
    )

    type_mapping = {
        0: "ORBITAL_DEVIATION",
        1: "VELOCITY_ANOMALY",
        2: "LIDAR_ANOMALY",
        3: "RADAR_ANOMALY",
        4: "RADAR_ANOMALY",
        5: "OPTICAL_ANOMALY"
    }

    contributors = []

    for index in ranked_indices[:3]:

        index = int(index)

        contributors.append({

            "feature":
                FEATURE_NAMES[index],

            "label":
                FEATURE_LABELS[index],

            "value":
                float(observation[index]),

            "unit":
                FEATURE_UNITS[index],

            "normal_noise_scale":
                float(NORMAL_SIGMA[index]),

            "sigma_multiple":
                float(deviations[index]),

            "threshold_exceeded":
                bool(
                    deviations[index]
                    > STATISTICAL_THRESHOLD
                )
        })

    return {

        "suspected_type":
            type_mapping[dominant],

        "dominant_feature":
            FEATURE_NAMES[dominant],

        "max_sigma_multiple":
            float(deviations[dominant]),

        "contributors":
            contributors,

        "explanation": (
            f"{FEATURE_LABELS[dominant]} "
            f"is {abs(observation[dominant]):.4f} "
            f"{FEATURE_UNITS[dominant]}, "
            f"approximately "
            f"{deviations[dominant]:.2f} times "
            f"the configured normal noise scale."
        ),

        "explanation_method":
            "Residual magnitude ranking"
    }


# ==========================================
# SEVERITY CLASSIFICATION
# ==========================================

def classify_severity(
    is_anomaly,
    sigma_multiple
):

    if not is_anomaly:
        return "NORMAL"

    if sigma_multiple >= 10:
        return "CRITICAL"

    if sigma_multiple >= 6:
        return "HIGH"

    if sigma_multiple >= 3:
        return "MEDIUM"

    return "LOW"


# ==========================================
# RECOMMENDED OPERATOR ACTION
# ==========================================

def recommended_action(anomaly_type):

    actions = {

        "ORBITAL_DEVIATION": (
            "Prioritise additional tracking "
            "and verify the orbital estimate."
        ),

        "VELOCITY_ANOMALY": (
            "Check for possible manoeuvres "
            "using independent observations."
        ),

        "LIDAR_ANOMALY": (
            "Verify LiDAR measurements "
            "and cross-check with radar."
        ),

        "RADAR_ANOMALY": (
            "Check radar measurement quality "
            "and cross-validate with LiDAR."
        ),

        "OPTICAL_ANOMALY": (
            "Verify optical pointing, "
            "calibration and visibility."
        )
    }

    return actions.get(
        anomaly_type,
        "Continue routine monitoring."
    )


# ==========================================
# SINGLE OBSERVATION ANALYSIS
# ==========================================

def analyze_observation(observation):

    observation = np.asarray(
        observation,
        dtype=float
    )

    if observation.shape != (
        len(FEATURE_NAMES),
    ):
        raise ValueError(
            "Exactly six features required"
        )

    detector = get_detector()

    details = detector.predict_details(
        observation
    )

    is_anomaly = bool(
        details["predictions"][0]
    )

    ml_detected = bool(
        details["ml_predictions"][0]
    )

    statistical_detected = bool(
        details["statistical_predictions"][0]
    )

    detection_source = get_detection_source(
        ml_detected,
        statistical_detected
    )

    explanation = explain_anomaly(
        observation
    )

    severity = classify_severity(
        is_anomaly,
        explanation["max_sigma_multiple"]
    )

    suspected_type = (
        explanation["suspected_type"]
        if is_anomaly
        else "NORMAL"
    )

    return {

        "is_anomaly":
            is_anomaly,

        "anomaly_score":
            float(details["scores"][0]),

        "ml_score":
            float(details["ml_scores"][0]),

        "statistical_score":
            float(
                details["statistical_scores"][0]
            ),

        "ml_detected":
            ml_detected,

        "statistical_detected":
            statistical_detected,

        "detection_source":
            detection_source,

        "severity":
            severity,

        "suspected_type":
            suspected_type,

        "features": {
            name: float(value)
            for name, value in zip(
                FEATURE_NAMES,
                observation
            )
        },

        "max_sigma_multiple":
            explanation["max_sigma_multiple"],

        "statistical_threshold":
            STATISTICAL_THRESHOLD,

        "explanation": (
            explanation["explanation"]
            if is_anomaly
            else "No anomaly detected."
        ),

        "contributors":
            explanation["contributors"],

        "explanation_method":
            explanation["explanation_method"],

        "recommended_action":
            recommended_action(suspected_type),

        "model":
            MODEL_NAME,

        "training_source":
            "Simulated normal residuals",

        "score_is_probability":
            False
    }


# ==========================================
# CALCULATE EVALUATION METRICS
# ==========================================

def calculate_metrics(
    labels,
    predictions
):

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    return {

        "accuracy": float(
            accuracy_score(
                labels,
                predictions
            )
        ),

        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0
            )
        ),

        "f1_score": float(
            f1_score(
                labels,
                predictions,
                zero_division=0
            )
        ),

        "confusion_matrix": {

            "true_negative":
                int(tn),

            "false_positive":
                int(fp),

            "false_negative":
                int(fn),

            "true_positive":
                int(tp)
        },

        "false_positive_rate": float(
            fp / max(tn + fp, 1)
        )
    }


# ==========================================
# MODEL EVALUATION
# ==========================================

def evaluate_model(
    samples=500,
    anomaly_fraction=0.10,
    strength=8.0,
    seed=123
):

    detector = get_detector()

    data, labels, types = (
        generate_evaluation_data(
            samples=samples,
            anomaly_fraction=anomaly_fraction,
            strength=strength,
            seed=seed
        )
    )

    details = detector.predict_details(
        data
    )

    predictions = details["predictions"]

    ml_predictions = (
        details["ml_predictions"]
    )

    statistical_predictions = (
        details["statistical_predictions"]
    )

    # Hybrid performance.

    hybrid_metrics = calculate_metrics(
        labels,
        predictions
    )

    # Original Isolation Forest performance.

    ml_metrics = calculate_metrics(
        labels,
        ml_predictions
    )

    # Statistical-only performance.

    statistical_metrics = calculate_metrics(
        labels,
        statistical_predictions
    )

    # Detection performance for each type.

    per_type_recall = {}

    ml_per_type_recall = {}

    types = np.array(types)

    for anomaly_type in ANOMALY_TYPES:

        mask = types == anomaly_type

        per_type_recall[anomaly_type] = (
            float(predictions[mask].mean())
            if mask.any()
            else None
        )

        ml_per_type_recall[anomaly_type] = (
            float(ml_predictions[mask].mean())
            if mask.any()
            else None
        )

    # Determine how many anomalies were
    # detected exclusively by each method.

    ml_mask = ml_predictions.astype(bool)

    statistical_mask = (
        statistical_predictions.astype(bool)
    )

    detection_counts = {

        "ml_only": int(
            np.sum(
                ml_mask
                & ~statistical_mask
            )
        ),

        "statistical_only": int(
            np.sum(
                statistical_mask
                & ~ml_mask
            )
        ),

        "both": int(
            np.sum(
                ml_mask
                & statistical_mask
            )
        )
    }

    return {

        "model":
            MODEL_NAME,

        "training_samples":
            TRAINING_SAMPLES,

        "evaluation_samples":
            samples,

        "injected_anomalies":
            int(labels.sum()),

        "detected_anomalies":
            int(predictions.sum()),

        **hybrid_metrics,

        "per_type_recall":
            per_type_recall,

        "ml_only_metrics":
            ml_metrics,

        "ml_only_per_type_recall":
            ml_per_type_recall,

        "statistical_only_metrics":
            statistical_metrics,

        "detection_source_counts":
            detection_counts,

        "recall_improvement_percentage_points":
            float(
                (
                    hybrid_metrics["recall"]
                    - ml_metrics["recall"]
                ) * 100
            ),

        "f1_improvement_percentage_points":
            float(
                (
                    hybrid_metrics["f1_score"]
                    - ml_metrics["f1_score"]
                ) * 100
            ),

        "statistical_threshold":
            STATISTICAL_THRESHOLD,

        "anomaly_strength":
            strength,

        "evaluation_seed":
            seed,

        "simulated":
            True
    }


# ==========================================
# SATELLITE ANOMALY DEMONSTRATION
# ==========================================

def run_anomaly_demo(
    norad_id=25544,
    samples=120,
    step_seconds=10,
    strength=8.0,
    start_time=None
):

    if samples < 30:
        raise ValueError(
            "At least 30 samples required"
        )

    if step_seconds <= 0:
        raise ValueError(
            "Step must be positive"
        )

    if not np.isfinite(strength) or strength <= 0:
        raise ValueError(
            "Strength must be positive and finite"
        )

    satellites = get_satellites()

    if norad_id not in satellites:
        raise KeyError(
            f"Unknown satellite: {norad_id}"
        )

    if start_time is None:

        start_time = datetime.now(
            timezone.utc
        )

    elif start_time.tzinfo is None:

        raise ValueError(
            "Start time must be timezone-aware"
        )

    metadata = satellites[norad_id][
        "metadata"
    ]

    satellite_name = metadata[
        "OBJECT_NAME"
    ]

    rng = np.random.default_rng(2026)

    data = generate_normal_data(
        samples=samples,
        seed=2027
    )

    labels = np.zeros(
        samples,
        dtype=int
    )

    types = ["NORMAL"] * samples

    # Inject five different anomaly episodes
    # at separate points in the timeline.

    for i, anomaly_type in enumerate(
        ANOMALY_TYPES
    ):

        start_index = int(
            samples * (i + 1) / 6
        )

        for index in range(
            start_index,
            min(start_index + 3, samples)
        ):

            data[index] = inject_anomaly(
                data[index],
                anomaly_type,
                rng,
                strength
            )

            labels[index] = 1

            types[index] = anomaly_type

    detector = get_detector()

    details = detector.predict_details(
        data
    )

    predictions = details["predictions"]

    scores = details["scores"]

    timeline = []

    alerts = []

    severity_counts = {
        "CRITICAL": 0,
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0
    }

    detection_source_counts = {
        "ML_AND_STATISTICAL": 0,
        "ISOLATION_FOREST": 0,
        "STATISTICAL_RESIDUAL": 0
    }

    # --------------------------------------
    # Generate timeline and explainable alerts
    # --------------------------------------

    for index in range(samples):

        timestamp = (
            start_time
            + timedelta(
                seconds=index * step_seconds
            )
        )

        is_anomaly = bool(
            predictions[index]
        )

        ml_detected = bool(
            details["ml_predictions"][index]
        )

        statistical_detected = bool(
            details["statistical_predictions"][
                index
            ]
        )

        detection_source = (
            get_detection_source(
                ml_detected,
                statistical_detected
            )
        )

        explanation = explain_anomaly(
            data[index]
        )

        severity = classify_severity(
            is_anomaly,
            explanation["max_sigma_multiple"]
        )

        suspected_type = (
            explanation["suspected_type"]
            if is_anomaly
            else "NORMAL"
        )

        record = {

            "sample":
                index,

            "timestamp":
                timestamp.isoformat(),

            "elapsed_seconds":
                index * step_seconds,

            "features": {
                name: float(value)
                for name, value in zip(
                    FEATURE_NAMES,
                    data[index]
                )
            },

            "anomaly_score":
                float(scores[index]),

            "ml_score":
                float(
                    details["ml_scores"][index]
                ),

            "statistical_score":
                float(
                    details["statistical_scores"][
                        index
                    ]
                ),

            "is_anomaly":
                is_anomaly,

            "ml_detected":
                ml_detected,

            "statistical_detected":
                statistical_detected,

            "detection_source":
                detection_source,

            "severity":
                severity,

            "suspected_type":
                suspected_type,

            "max_sigma_multiple":
                explanation["max_sigma_multiple"],

            "ground_truth_anomaly":
                bool(labels[index]),

            "ground_truth_type":
                types[index]
        }

        timeline.append(record)

        if is_anomaly:

            # Associate the simulated anomaly
            # with a real catalogue-predicted
            # satellite position.

            state = get_state(
                norad_id,
                timestamp
            )

            alerts.append({

                **record,

                "alert_id": (
                    f"ANO-{norad_id}-{index}"
                ),

                "norad_id":
                    norad_id,

                "satellite_name":
                    satellite_name,

                "position_km": np.asarray(
                    state["position_km"]
                ).tolist(),

                "explanation":
                    explanation["explanation"],

                "contributors":
                    explanation["contributors"],

                "explanation_method":
                    explanation["explanation_method"],

                "recommended_action":
                    recommended_action(
                        suspected_type
                    ),

                "simulated":
                    True
            })

            severity_counts[severity] += 1

            detection_source_counts[
                detection_source
            ] += 1

    metrics = calculate_metrics(
        labels,
        predictions
    )

    ml_metrics = calculate_metrics(
        labels,
        details["ml_predictions"]
    )

    return {

        "mode":
            "synthetic_anomaly_demo",

        "model":
            MODEL_NAME,

        "simulated":
            True,

        "norad_id":
            norad_id,

        "satellite_name":
            satellite_name,

        "total_samples":
            samples,

        "injected_anomalies":
            int(labels.sum()),

        "detected_anomalies":
            int(predictions.sum()),

        "normal_samples":
            int(samples - predictions.sum()),

        "severity_counts":
            severity_counts,

        "detection_source_counts":
            detection_source_counts,

        **metrics,

        "ml_only_metrics":
            ml_metrics,

        "statistical_threshold":
            STATISTICAL_THRESHOLD,

        "score_is_probability":
            False,

        "timeline":
            timeline,

        "alerts":
            alerts
    }


# ==========================================
# MODULE TEST
# ==========================================

if __name__ == "__main__":

    import json

    print(
        "COSMOTRACK AI ANOMALY ENGINE"
    )

    print("=" * 55)

    print(
        "\nMODEL: HYBRID ANOMALY DETECTION"
    )

    print(
        "Isolation Forest + "
        "Statistical Residual Analysis"
    )

    print("\nMODEL EVALUATION")

    metrics = evaluate_model()

    print(json.dumps(
        metrics,
        indent=2
    ))

    # --------------------------------------
    # Performance comparison
    # --------------------------------------

    print("\n" + "=" * 65)

    print("ML-ONLY VS HYBRID PERFORMANCE")

    print("=" * 65)

    print(
        f"{'Metric':<25}"
        f"{'ML Only':>15}"
        f"{'Hybrid':>15}"
    )

    print("-" * 65)

    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate"
    ]:

        ml_value = metrics[
            "ml_only_metrics"
        ][key]

        hybrid_value = metrics[key]

        print(
            f"{key:<25}"
            f"{ml_value:>14.2%}"
            f"{hybrid_value:>14.2%}"
        )

    print("\nPER-TYPE DETECTION PERFORMANCE")

    print("-" * 65)

    for anomaly_type in ANOMALY_TYPES:

        ml_recall = metrics[
            "ml_only_per_type_recall"
        ][anomaly_type]

        hybrid_recall = metrics[
            "per_type_recall"
        ][anomaly_type]

        print(
            f"{anomaly_type:<25}"
            f"{ml_recall:>14.2%}"
            f"{hybrid_recall:>14.2%}"
        )

    # --------------------------------------
    # Satellite anomaly demonstration
    # --------------------------------------

    print("\n" + "=" * 55)

    print("SATELLITE ANOMALY DEMO")

    print("=" * 55)

    demo = run_anomaly_demo()

    print(
        "Satellite:",
        demo["satellite_name"]
    )

    print(
        "Total samples:",
        demo["total_samples"]
    )

    print(
        "Injected anomalies:",
        demo["injected_anomalies"]
    )

    print(
        "Detected anomalies:",
        demo["detected_anomalies"]
    )

    print(
        "Severity counts:",
        demo["severity_counts"]
    )

    print(
        "Detection sources:",
        demo["detection_source_counts"]
    )

    if demo["alerts"]:

        print("\nSAMPLE EXPLAINABLE ALERT")

        print(json.dumps(
            demo["alerts"][0],
            indent=2
        ))

    print(
        "\nAnomaly engine completed!"
    )
