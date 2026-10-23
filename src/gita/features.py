
from pathlib import Path

import numpy as np
import pandas as pd


def build_order_features(data_dir: Path) -> pd.DataFrame:
    """Build one numerical feature row per order from Olist datasets."""

    data_dir = Path(data_dir)

    orders_path = data_dir / "olist_orders_dataset.csv"
    items_path = data_dir / "olist_order_items_dataset.csv"
    payments_path = data_dir / "olist_order_payments_dataset.csv"

    # 1. Check that all required datasets exist.
    required_files = [orders_path, items_path, payments_path]

    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required dataset not found: {file_path}"
            )

    # 2. Load only the columns required for feature engineering.
    orders = pd.read_csv(
        orders_path,
        usecols=["order_id", "order_status"],
    )

    items = pd.read_csv(
        items_path,
        usecols=["order_id", "price", "freight_value"],
    )

    payments = pd.read_csv(
        payments_path,
        usecols=["order_id", "payment_value"],
    )

    # 3. Validate order identifiers.
    if orders["order_id"].isna().any():
        raise ValueError("Orders contain missing order IDs.")

    if orders["order_id"].duplicated().any():
        raise ValueError("Orders contain duplicate order IDs.")

    if items["order_id"].isna().any():
        raise ValueError("Order items contain missing order IDs.")

    if payments["order_id"].isna().any():
        raise ValueError("Payments contain missing order IDs.")

    # 4. Convert transaction values to numeric.
    for column in ["price", "freight_value"]:
        items[column] = pd.to_numeric(
            items[column],
            errors="raise",
        )

    payments["payment_value"] = pd.to_numeric(
        payments["payment_value"],
        errors="raise",
    )

    # 5. Reject missing, infinite, or invalid transaction values.
    for column in ["price", "freight_value"]:
        values = items[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Invalid or non-finite values in items.{column}"
            )

    payment_values = payments["payment_value"].to_numpy(dtype=float)

    if not np.isfinite(payment_values).all():
        raise ValueError(
            "Invalid or non-finite values in payments.payment_value"
        )

    # 6. Aggregate item records to one row per order.
    item_totals = (
        items.groupby("order_id", as_index=False)
        .agg(
            item_sales=("price", "sum"),
            freight=("freight_value", "sum"),
            item_records=("order_id", "size"),
        )
    )

    # 7. Aggregate payment records separately.
    payment_totals = (
        payments.groupby("order_id", as_index=False)
        .agg(
            payment_total=("payment_value", "sum"),
            payment_records=("order_id", "size"),
        )
    )

    # 8. Join aggregated tables to the unique orders table.
    features = orders.merge(
        item_totals,
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    features = features.merge(
        payment_totals,
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    # 9. Preserve missing-record information.
    features["has_item_records"] = (
        features["item_records"].notna().astype(int)
    )

    features["has_payment_records"] = (
        features["payment_records"].notna().astype(int)
    )

    # 10. Fill missing aggregates while preserving presence flags.
    numeric_columns = [
        "item_sales",
        "freight",
        "item_records",
        "payment_total",
        "payment_records",
    ]

    features[numeric_columns] = features[numeric_columns].fillna(0)

    # 11. Calculate expected totals and reconciliation differences.
    features["expected_total"] = (
        features["item_sales"] + features["freight"]
    )

    features["difference"] = (
        features["payment_total"] - features["expected_total"]
    )

    # 12. Calculate derived business features.
    # NaN represents an undefined ratio or average, not a zero value.
    features["freight_ratio"] = (
        features["freight"]
        .div(features["item_sales"].where(features["item_sales"] != 0))
    )

    features["average_item_value"] = (
        features["item_sales"]
        .div(features["item_records"].where(features["item_records"] != 0))
    )

    # 13. Confirm that each order appears exactly once.
    if features["order_id"].duplicated().any():
        raise ValueError(
            "Feature table contains duplicate order IDs."
        )

    if len(features) != len(orders):
        raise ValueError(
            "Feature table row count does not match the orders table."
        )

    # 14. Validate core financial values.
    # Derived features may contain NaN when their denominator is zero.
    core_numeric_columns = [
        "item_sales",
        "freight",
        "payment_total",
        "item_records",
        "payment_records",
        "expected_total",
        "difference",
    ]

    core_values = features[core_numeric_columns].to_numpy(dtype=float)

    if not np.isfinite(core_values).all():
        raise ValueError(
            "Non-finite values found in the core feature table."
        )

    # Any defined derived feature must be finite.
    for column in ["freight_ratio", "average_item_value"]:
        defined_values = features[column].dropna().to_numpy(dtype=float)

        if not np.isfinite(defined_values).all():
            raise ValueError(
                f"Non-finite values found in {column}."
            )

    # 15. Attach a summary for diagnostics and validation.
    features.attrs["quality_summary"] = {
        "total_orders": len(features),
        "orders_without_items": int(
            (features["has_item_records"] == 0).sum()
        ),
        "orders_without_payments": int(
            (features["has_payment_records"] == 0).sum()
        ),
        "orders_with_large_differences": int(
            (features["difference"].abs() > 1).sum()
        ),
        "orders_with_undefined_freight_ratio": int(
            features["freight_ratio"].isna().sum()
        ),
        "orders_with_undefined_average_item_value": int(
            features["average_item_value"].isna().sum()
        ),
    }

    return features
