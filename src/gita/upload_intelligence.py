
"""Gita anomaly detection for user-uploaded business datasets.

This is a separate generic path. It does not use the Olist-trained model.
It fits an exploratory Isolation Forest to the selected numeric columns
of the uploaded dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


MIN_ROWS = 30
MAX_FEATURES = 30
RANDOM_STATE = 42


def _suggest_numeric_features(df: pd.DataFrame) -> list[str]:
    """Suggest numeric columns while excluding likely identifiers."""
    excluded_terms = (
        "id",
        "zip",
        "postal",
        "phone",
        "code",
        "postcode",
    )

    candidates = []

    for column in df.select_dtypes(include="number").columns:
        name = str(column).strip().lower()

        if any(term in name for term in excluded_terms):
            continue

        values = pd.to_numeric(df[column], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)

        # A feature must have at least two distinct valid values.
        if values.notna().sum() >= 2 and values.nunique(dropna=True) > 1:
            candidates.append(column)

    return candidates[:MAX_FEATURES]


def analyze_uploaded_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict]:
    """Fit an exploratory Isolation Forest and score uploaded rows."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected the uploaded data as a pandas DataFrame.")

    if df.empty:
        raise ValueError("The uploaded dataset has no rows.")

    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Gita needs at least {MIN_ROWS} rows for this exploratory analysis. "
            f"This dataset has {len(df)} rows."
        )

    if not feature_columns:
        raise ValueError("Select at least one numeric feature.")

    missing_columns = [
        column for column in feature_columns if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Selected columns are missing: {missing_columns}")

    X = df[feature_columns].copy()

    for column in feature_columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)

    # Do not silently accept columns that contain no usable values.
    all_missing = [column for column in X.columns if X[column].notna().sum() == 0]
    if all_missing:
        raise ValueError(
            "These selected columns contain no usable numeric values: "
            + ", ".join(map(str, all_missing))
        )

    # Remove constant columns: they cannot help distinguish records.
    constant_columns = [
        column
        for column in X.columns
        if X[column].nunique(dropna=True) <= 1
    ]
    if constant_columns:
        X = X.drop(columns=constant_columns)

    if X.shape[1] == 0:
        raise ValueError(
            "No informative numeric columns remain. Select columns with "
            "more than one distinct value."
        )

    # Isolation Forest requires finite input. Median imputation is a
    # practical baseline; the report also shows missing-value counts.
    missing_counts = X.isna().sum()
    imputer = SimpleImputer(strategy="median")
    X_ready = imputer.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    predictions = model.fit_predict(X_ready)
    scores = model.score_samples(X_ready)

    results = df.copy()
    results.insert(0, "gita_row_number", np.arange(1, len(df) + 1))
    results["gita_anomaly_score"] = scores
    results["gita_is_anomaly"] = predictions == -1
    results["gita_result"] = np.where(
        predictions == -1,
        "Review recommended",
        "No anomaly flagged",
    )

    # Lower score means more unusual relative to this uploaded dataset.
    results = results.sort_values(
        "gita_anomaly_score",
        ascending=True,
    ).reset_index(drop=True)

    flagged_count = int((predictions == -1).sum())

    summary = {
        "rows_analyzed": int(len(df)),
        "features_used": list(X.columns),
        "features_dropped_as_constant": list(constant_columns),
        "missing_values_imputed": {
            str(column): int(count)
            for column, count in missing_counts.items()
            if count > 0
        },
        "anomalies_flagged": flagged_count,
        "anomaly_rate_percent": round(
            flagged_count / len(df) * 100, 2
        ),
        "model": "IsolationForest",
        "model_scope": "Fitted to this uploaded dataset only",
    }

    return results, summary


def render_uploaded_gita_tab(df: pd.DataFrame) -> None:
    """Render Gita's generic uploaded-data analysis in a Streamlit page."""

    st.subheader("Gita Intelligence — Uploaded Dataset")

    st.caption(
        "Exploratory anomaly detection on selected numeric columns. "
        "A flagged row is unusual relative to this dataset; it is not "
        "proof of fraud, an error, or financial loss."
    )

    if df is None:
        st.info("Upload a CSV or XLSX file and pass its DataFrame to Gita.")
        return

    if not isinstance(df, pd.DataFrame) or df.empty:
        st.warning("The uploaded dataset is empty or could not be read.")
        return

    st.write(f"Dataset size: **{len(df):,} rows** × **{len(df.columns):,} columns")

    numeric_columns = _suggest_numeric_features(df)

    if not numeric_columns:
        st.warning(
            "No suitable numeric features were detected. "
            "Gita's generic Isolation Forest path currently requires "
            "numeric business fields."
        )
        return

    selected_features = st.multiselect(
        "Choose numeric fields for anomaly detection",
        options=numeric_columns,
        default=numeric_columns[: min(5, len(numeric_columns))],
        help=(
            "Choose meaningful measures such as sales, revenue, quantity, "
            "unit price, costs, or margin. Avoid customer IDs, order IDs, "
            "postal codes, and other identifiers."
        ),
        key="gita_uploaded_feature_columns",
    )

    st.caption(
        "Gita will fit a new exploratory model to this upload. "
        "Results are dataset-relative and are not a validated accuracy score."
    )

    if st.button(
        "Analyze uploaded data with Gita",
        type="primary",
        key="run_gita_uploaded_analysis",
    ):
        try:
            with st.spinner("Gita is analyzing the selected numeric features..."):
                results, summary = analyze_uploaded_dataset(
                    df,
                    selected_features,
                )

            st.session_state["gita_uploaded_results"] = results
            st.session_state["gita_uploaded_summary"] = summary
            st.session_state["gita_uploaded_features"] = selected_features

        except (ValueError, TypeError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(
                "Gita could not analyze this dataset. "
                f"Technical detail: {exc}"
            )

    results = st.session_state.get("gita_uploaded_results")
    summary = st.session_state.get("gita_uploaded_summary")
    previous_features = st.session_state.get("gita_uploaded_features")

    # Do not display old results as if they came from a newly selected mapping.
    if previous_features != selected_features:
        results = None
        summary = None

    if results is None or summary is None:
        st.info(
            f"Select features and run the analysis. At least {MIN_ROWS} rows "
            "are required for this exploratory version."
        )
        return

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Rows analyzed", f"{summary['rows_analyzed']:,}")
    metric2.metric("Rows flagged", f"{summary['anomalies_flagged']:,}")
    metric3.metric("Flagged share", f"{summary['anomaly_rate_percent']:.2f}%")

    st.markdown("#### What Gita analyzed")
    st.write(", ".join(summary["features_used"]))

    if summary["features_dropped_as_constant"]:
        st.caption(
            "Constant columns excluded: "
            + ", ".join(summary["features_dropped_as_constant"])
        )

    if summary["missing_values_imputed"]:
        st.warning(
            "Missing numeric values were median-imputed for model input. "
            "Review data quality before acting on results."
        )
        st.json(summary["missing_values_imputed"])

    st.markdown("#### Records to investigate first")
    flagged = results[results["gita_is_anomaly"]]

    if flagged.empty:
        st.success("No rows were flagged by this model run.")
    else:
        display_columns = [
            "gita_row_number",
            *summary["features_used"],
            "gita_anomaly_score",
            "gita_result",
        ]
        st.dataframe(
            flagged[display_columns],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("View all scored records"):
        st.dataframe(results, use_container_width=True, hide_index=True)

    csv_data = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Gita results as CSV",
        data=csv_data,
        file_name="gita_uploaded_dataset_results.csv",
        mime="text/csv",
        key="download_gita_uploaded_results",
    )
