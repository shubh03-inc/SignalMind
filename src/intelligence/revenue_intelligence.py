
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"

DIFFERENCE_THRESHOLD = 1.00


def _read_csv(filename, required_columns):
    """Load a CSV file and verify its required columns."""
    path = DATA / filename

    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")

    frame = pd.read_csv(path)

    missing_columns = set(required_columns) - set(frame.columns)

    if missing_columns:
        raise ValueError(
            f"{filename} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return frame[required_columns].copy()


def _validate_order_ids(frame, source_name):
    """Reject missing order IDs."""
    if frame["order_id"].isna().any():
        raise ValueError(
            f"{source_name} contains missing order IDs."
        )


def _validate_amounts(frame, columns, source_name):
    """Require finite, numeric, non-negative monetary values."""
    for column in columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        values = frame[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"{source_name} contains invalid or non-finite "
                f"values in '{column}'."
            )

        if (frame[column] < 0).any():
            raise ValueError(
                f"{source_name} contains negative values in '{column}'."
            )


def _describe_issues(row):
    """Describe every applicable reconciliation issue for an order."""
    issues = []

    if row["missing_items"]:
        issues.append("No item records")

    if row["missing_payments"]:
        issues.append("No payment records")

    if row["large_difference"]:
        issues.append("Payment difference exceeds threshold")

    return "; ".join(issues) if issues else "None"


def build_revenue_intelligence():
    """
    Reconcile item sales and freight against recorded payments.

    Returns order-level results, exception tables, and summary tables.
    """

    # 1. Load source data.
    orders = _read_csv(
        "olist_orders_dataset.csv",
        ["order_id", "order_status"],
    )

    items = _read_csv(
        "olist_order_items_dataset.csv",
        ["order_id", "price", "freight_value"],
    )

    payments = _read_csv(
        "olist_order_payments_dataset.csv",
        ["order_id", "payment_value"],
    )

    # 2. Validate the orders table.
    if orders["order_id"].isna().any():
        raise ValueError("Orders contain missing order IDs.")

    if orders["order_id"].duplicated().any():
        duplicates = (
            orders.loc[
                orders["order_id"].duplicated(keep=False),
                "order_id",
            ]
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"Orders contain duplicate order IDs: {duplicates}"
        )

    # 3. Validate IDs in the item and payment tables.
    _validate_order_ids(items, "Order items")
    _validate_order_ids(payments, "Payments")

    # 4. Reject records that refer to orders absent from the orders table.
    known_order_ids = set(orders["order_id"])

    unknown_item_ids = set(items["order_id"]) - known_order_ids
    unknown_payment_ids = set(payments["order_id"]) - known_order_ids

    if unknown_item_ids:
        raise ValueError(
            "Order items contain unknown order IDs: "
            f"{sorted(unknown_item_ids)[:10]}"
        )

    if unknown_payment_ids:
        raise ValueError(
            "Payments contain unknown order IDs: "
            f"{sorted(unknown_payment_ids)[:10]}"
        )

    # 5. Validate monetary columns.
    _validate_amounts(
        items,
        ["price", "freight_value"],
        "Order items",
    )

    _validate_amounts(
        payments,
        ["payment_value"],
        "Payments",
    )

    # 6. Aggregate all item rows to one row per order.
    item_summary = (
        items.groupby("order_id", as_index=False)
        .agg(
            item_sales=("price", "sum"),
            freight=("freight_value", "sum"),
            item_records=("order_id", "size"),
        )
    )

    # 7. Aggregate all payment rows to one row per order.
    payment_summary = (
        payments.groupby("order_id", as_index=False)
        .agg(
            payment_total=("payment_value", "sum"),
            payment_records=("order_id", "size"),
        )
    )

    # 8. Merge the aggregates into the orders table.
    check = orders.merge(
        item_summary,
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    check = check.merge(
        payment_summary,
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    # 9. Replace absent aggregate values with zero.
    numeric_columns = [
        "item_sales",
        "freight",
        "item_records",
        "payment_total",
        "payment_records",
    ]

    check[numeric_columns] = check[numeric_columns].fillna(0)

    check["item_records"] = check["item_records"].astype(int)
    check["payment_records"] = check["payment_records"].astype(int)

    # 10. Calculate reconciliation differences.
    check["item_plus_freight"] = (
        check["item_sales"] + check["freight"]
    )

    check["difference"] = (
        check["payment_total"] - check["item_plus_freight"]
    )

    check["absolute_difference"] = check["difference"].abs()

    # 11. Flag missing records and significant differences independently.
    no_items = check["item_records"].eq(0)
    no_payments = check["payment_records"].eq(0)

    large_difference = (
        check["absolute_difference"] > DIFFERENCE_THRESHOLD
    )

    check["missing_items"] = no_items
    check["missing_payments"] = no_payments
    check["large_difference"] = large_difference

    # Keep the issue descriptions expected by existing validation tests.
    check["issue"] = check.apply(_describe_issues, axis=1)

    # 12. Assign one primary status to each order.
    # Priority: missing items, missing payments, amount mismatch, reconciled.
    check["reconciliation_status"] = "RECONCILED"

    check.loc[
        large_difference,
        "reconciliation_status",
    ] = "AMOUNT_MISMATCH"

    check.loc[
        no_payments,
        "reconciliation_status",
    ] = "MISSING_PAYMENTS"

    check.loc[
        no_items,
        "reconciliation_status",
    ] = "MISSING_ITEMS"

    # 13. Summarize totals by order status.
    status_summary = (
        check.groupby("order_status", dropna=False)
        .agg(
            order_count=("order_id", "nunique"),
            item_sales=("item_sales", "sum"),
            freight=("freight", "sum"),
            payments=("payment_total", "sum"),
            item_plus_freight=("item_plus_freight", "sum"),
            payment_difference=("difference", "sum"),
        )
        .reset_index()
    )

    # 14. Summarize orders by their primary reconciliation status.
    reconciliation_summary = (
        check.groupby("reconciliation_status", dropna=False)
        .agg(
            order_count=("order_id", "nunique"),
            item_sales=("item_sales", "sum"),
            freight=("freight", "sum"),
            payments=("payment_total", "sum"),
            payment_difference=("difference", "sum"),
        )
        .reset_index()
    )

    # 15. Create investigation tables.
    mismatches = check.loc[
        no_items | no_payments | large_difference
    ].copy()

    missing_items_with_payments = check.loc[
        no_items & ~no_payments
    ].copy()

    delivered_differences = check.loc[
        check["order_status"].eq("delivered") & large_difference
    ].copy()

    # 16. Return the outputs for use by SignalMind's reporting layer.
    return {
        "reconciliation": check,
        "status_summary": status_summary,
        "reconciliation_summary": reconciliation_summary,
        "mismatches": mismatches,
        "missing_items_with_payments": missing_items_with_payments,
        "delivered_differences": delivered_differences,
        "total_orders": len(check),
        "orders_with_items": int((~no_items).sum()),
        "orders_without_items": int(no_items.sum()),
        "orders_with_payments": int((~no_payments).sum()),
        "orders_without_payments": int(no_payments.sum()),
        "orders_with_large_differences": int(large_difference.sum()),
        "delivered_orders_with_large_differences": len(
            delivered_differences
        ),
    }


if __name__ == "__main__":
    result = build_revenue_intelligence()

    print("=" * 60)
    print("SIGNALMIND REVENUE INTELLIGENCE")
    print("=" * 60)

    print(f"Total orders: {result['total_orders']}")
    print(f"Orders with items: {result['orders_with_items']}")
    print(f"Orders without items: {result['orders_without_items']}")
    print(f"Orders with payments: {result['orders_with_payments']}")
    print(f"Orders without payments: {result['orders_without_payments']}")
    print(
        "Orders with large differences: "
        f"{result['orders_with_large_differences']}"
    )
    print(
        "Delivered orders with large differences: "
        f"{result['delivered_orders_with_large_differences']}"
    )

    print("\nRECONCILIATION SUMMARY")
    print(result["reconciliation_summary"].to_string(index=False))

    print("\nSUMMARY BY ORDER STATUS")
    print(result["status_summary"].to_string(index=False))
