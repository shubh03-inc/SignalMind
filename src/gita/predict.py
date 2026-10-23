from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.gita.features import build_order_features


# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT / "data"
MODEL_PATH = PROJECT / "models" / "gita_isolation_forest.joblib"
REPORT_DIR = PROJECT / "reports"

PREDICTION_PATH = REPORT_DIR / "gita_predictions.csv"


# ==================================================
# LOAD AND VALIDATE MODEL
# ==================================================

def load_gita():
    """Load the saved Gita model and validate its artifact."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Saved Gita model not found:\n{MODEL_PATH}\n"
            "Please verify that the model has been trained."
        )

    artifact = joblib.load(MODEL_PATH)

    required_keys = {
        "model",
        "scaler",
        "feature_columns",
        "model_version",
    }

    if not isinstance(artifact, dict):
        raise TypeError(
            "Invalid model artifact: expected a dictionary."
        )

    missing_keys = required_keys - set(artifact.keys())

    if missing_keys:
        raise ValueError(
            "Model artifact is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    feature_columns = artifact["feature_columns"]

    if (
        not isinstance(feature_columns, list)
        or not feature_columns
        or not all(isinstance(col, str) for col in feature_columns)
    ):
        raise ValueError(
            "Model artifact contains invalid feature columns."
        )

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError(
            "Model artifact contains duplicate feature columns."
        )

    model = artifact["model"]
    scaler = artifact["scaler"]

    if not hasattr(model, "predict"):
        raise TypeError(
            "Saved model does not support prediction."
        )

    if not hasattr(model, "decision_function"):
        raise TypeError(
            "Saved model does not support anomaly scoring."
        )

    if not hasattr(scaler, "transform"):
        raise TypeError(
            "Saved scaler does not support transformation."
        )

    if hasattr(scaler, "n_features_in_"):
        if scaler.n_features_in_ != len(feature_columns):
            raise ValueError(
                "Feature count mismatch between the saved scaler "
                "and the model artifact."
            )

    return artifact


# ==================================================
# PREDICTION FUNCTION
# ==================================================

def predict_gita(data=None, save_report=False):
    """
    Score eligible orders using the saved Gita model.

    Parameters
    ----------
    data : pandas.DataFrame or None
        Order-level features. When None, features are generated
        from the configured project data files.

    save_report : bool
        Save eligible-order predictions to a CSV file when True.

    Returns
    -------
    pandas.DataFrame
        Eligible orders with Gita's predictions and anomaly scores.

    Notes
    -----
    Isolation Forest returns:
        -1 = anomaly
         1 = inlier

    Lower decision_function scores indicate more unusual
    observations. Scores are not probabilities.
    """

    # --------------------------------------------------
    # 1. Load saved model
    # --------------------------------------------------

    artifact = load_gita()

    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_columns = artifact["feature_columns"]

    # --------------------------------------------------
    # 2. Build or validate input data
    # --------------------------------------------------

    if data is None:
        data = build_order_features(DATA_DIR)

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Input data must be a pandas DataFrame or None."
        )

    if data.empty:
        raise ValueError(
            "No order data is available for prediction."
        )

    if not data.columns.is_unique:
        raise ValueError(
            "Input data contains duplicate column names."
        )

    # Do not modify the caller's original DataFrame.
    result = data.copy()

    # --------------------------------------------------
    # 3. Match training eligibility
    # --------------------------------------------------

    if "has_item_records" not in result.columns:
        raise ValueError(
            "Input data is missing 'has_item_records'. "
            "This feature is required to apply the training "
            "eligibility rule."
        )

    # Ensure eligibility values are valid.
    eligibility = pd.to_numeric(
        result["has_item_records"],
        errors="coerce",
    )

    invalid_eligibility = (
        eligibility.isna()
        | ~eligibility.isin([0, 1])
    )

    if invalid_eligibility.any():
        raise ValueError(
            "'has_item_records' must contain only 0 or 1 "
            "with no missing values."
        )

    eligible_mask = eligibility.eq(1)

    excluded_count = int((~eligible_mask).sum())

    result = result.loc[eligible_mask].copy()

    if result.empty:
        raise ValueError(
            "No eligible orders were found. Gita v0.2 scores "
            "orders that have item records."
        )

    print(
        "Orders excluded because they have no item records: "
        f"{excluded_count:,}"
    )

    # --------------------------------------------------
    # 4. Validate required model features
    # --------------------------------------------------

    missing_columns = [
        column
        for column in feature_columns
        if column not in result.columns
    ]

    if missing_columns:
        raise ValueError(
            "Prediction data is missing required features: "
            f"{missing_columns}"
        )

    # Preserve the exact column order from training.
    X = result.loc[:, feature_columns].copy()

    # Convert features to numeric.
    for column in feature_columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="raise",
        )

    # Reject missing and infinite values.
    X_values = X.to_numpy(dtype=float)

    invalid_mask = ~np.isfinite(X_values)

    if invalid_mask.any():
        invalid_counts = {}

        for index, column in enumerate(feature_columns):
            count = int(invalid_mask[:, index].sum())

            if count:
                invalid_counts[column] = count

        raise ValueError(
            "Eligible prediction rows contain missing or "
            "infinite feature values. No predictions were "
            "generated. Invalid counts: "
            f"{invalid_counts}"
        )

    # --------------------------------------------------
    # 5. Apply the saved scaler
    # --------------------------------------------------

    X_scaled = scaler.transform(X)

    if not np.isfinite(X_scaled).all():
        raise ValueError(
            "The saved scaler produced missing or infinite values."
        )

    # --------------------------------------------------
    # 6. Generate anomaly predictions
    # --------------------------------------------------

    predictions = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)

    if not np.isfinite(scores).all():
        raise ValueError(
            "The model produced non-finite anomaly scores."
        )

    # Keep the original financial columns and add model outputs.
    result["gita_prediction"] = predictions

    result["gita_anomaly_score"] = scores

    result["gita_is_anomaly"] = (
        predictions == -1
    )

    result["gita_result"] = np.where(
        result["gita_is_anomaly"],
        "Unusual pattern - review recommended",
        "No anomaly flagged",
    )

    result["gita_model_version"] = (
        artifact["model_version"]
    )

    # --------------------------------------------------
    # 7. Sort most unusual observations first
    # --------------------------------------------------

    result = result.sort_values(
        "gita_anomaly_score",
        ascending=True,
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 8. Optionally save the report
    # --------------------------------------------------

    if save_report:
        REPORT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        result.to_csv(
            PREDICTION_PATH,
            index=False,
        )

    return result


# ==================================================
# SCRIPT ENTRY POINT
# ==================================================

if __name__ == "__main__":

    try:
        predictions = predict_gita(
            data=None,
            save_report=True,
        )

        total_orders = len(predictions)

        anomaly_count = int(
            predictions["gita_is_anomaly"].sum()
        )

        inlier_count = total_orders - anomaly_count

        anomaly_rate = (
            anomaly_count / total_orders * 100
            if total_orders > 0
            else 0.0
        )

        print("\n" + "=" * 55)
        print("GITA PREDICTION SUMMARY")
        print("=" * 55)

        print(
            f"Model version: "
            f"{predictions['gita_model_version'].iloc[0]}"
        )

        print(f"Eligible orders scored: {total_orders:,}")
        print(f"Unusual patterns flagged: {anomaly_count:,}")
        print(f"No anomaly flagged: {inlier_count:,}")
        print(f"Anomaly rate: {anomaly_rate:.2f}%")

        print(f"\nPrediction report:\n{PREDICTION_PATH}")

        display_columns = [
            column
            for column in [
                "order_id",
                "order_status",
                "item_sales",
                "freight",
                "payment_total",
                "difference",
                "gita_anomaly_score",
                "gita_result",
            ]
            if column in predictions.columns
        ]

        print("\nFive most unusual orders")
        print("-" * 55)

        print(
            predictions[display_columns]
            .head(5)
            .to_string(index=False)
        )

        print("\nPrediction completed successfully.")

    except (ValueError, TypeError, FileNotFoundError) as error:
        raise SystemExit(
            f"\nGita prediction failed:\n{error}"
        ) from error
