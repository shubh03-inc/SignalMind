from pathlib import Path
import sys

import pandas as pd


# --------------------------------------------------
# PROJECT PATHS AND IMPORTS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from intelligence.sales_intelligence import build_sales_intelligence


# --------------------------------------------------
# MONTHLY ORDER ACTIVITY
# --------------------------------------------------

def build_monthly_order_activity(data_dir: Path) -> tuple[pd.DataFrame, int]:
    """
    Build a monthly summary of order activity by order status.

    Returns:
        monthly_activity: Monthly item/order counts grouped by status.
        unmatched_items: Number of item records without a matching order.
    """

    orders_path = data_dir / "olist_orders_dataset.csv"
    items_path = data_dir / "olist_order_items_dataset.csv"

    required_order_columns = {
        "order_id",
        "order_status",
        "order_purchase_timestamp",
    }
    required_item_columns = {"order_id"}

    # Check that the required input files exist.
    if not orders_path.is_file():
        raise FileNotFoundError(
            f"Orders dataset not found: {orders_path}"
        )

    if not items_path.is_file():
        raise FileNotFoundError(
            f"Order items dataset not found: {items_path}"
        )

    # Load the datasets.
    orders = pd.read_csv(orders_path)
    items = pd.read_csv(items_path)

    # Validate the columns before using them.
    missing_order_columns = required_order_columns.difference(
        orders.columns
    )

    if missing_order_columns:
        raise ValueError(
            "Orders dataset is missing required columns: "
            + ", ".join(sorted(missing_order_columns))
        )

    missing_item_columns = required_item_columns.difference(
        items.columns
    )

    if missing_item_columns:
        raise ValueError(
            "Items dataset is missing required columns: "
            + ", ".join(sorted(missing_item_columns))
        )

    # Keep only the columns needed for this report.
    orders = orders[
        [
            "order_id",
            "order_status",
            "order_purchase_timestamp",
        ]
    ].copy()

    items = items[["order_id"]].copy()

    # Convert purchase timestamps to datetime values.
    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce",
    )

    # Validate order IDs.
    if orders["order_id"].isna().any():
        raise ValueError(
            "Order records contain missing order IDs."
        )

    if orders["order_id"].duplicated().any():
        raise ValueError(
            "Duplicate order IDs found in the orders dataset."
        )

    if items["order_id"].isna().any():
        raise ValueError(
            "Item records contain missing order IDs."
        )

    # Attach order status and purchase date to each item record.
    item_data = items.merge(
        orders,
        on="order_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    # Count item records that do not match an order.
    unmatched_items = int(
        item_data["_merge"].eq("left_only").sum()
    )

    item_data = item_data.drop(columns=["_merge"])

    # The left join must preserve the original number of item rows.
    if len(item_data) != len(items):
        raise ValueError(
            "The order join unexpectedly changed the item row count."
        )

    # Derive the purchase month.
    item_data["purchase_month"] = (
        item_data["order_purchase_timestamp"].dt.to_period("M")
    )

    # Count item rows and distinct orders for each month and status.
    monthly_activity = (
        item_data.groupby(
            ["purchase_month", "order_status"],
            dropna=False,
        )
        .agg(
            item_rows=("order_id", "size"),
            order_count=("order_id", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["purchase_month", "order_status"],
            na_position="last",
        )
    )

    # Convert monthly periods into readable strings for CSV output.
    monthly_activity["purchase_month"] = (
        monthly_activity["purchase_month"].astype(str)
    )

    return monthly_activity, unmatched_items


# --------------------------------------------------
# MAIN SALES REPORT
# --------------------------------------------------

def main() -> None:
    """Generate SignalMind's sales report files."""

    data_dir = PROJECT_DIR / "data"
    reports_dir = PROJECT_DIR / "reports"

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Build sales metrics using the existing intelligence engine.
    result = build_sales_intelligence()

    # Validate the structure returned by the intelligence engine.
    required_result_keys = {
        "total_item_records",
        "total_delivered_orders",
        "orders_with_items",
        "orders_without_items",
        "item_sales",
        "freight_total",
        "average_order_value",
        "average_item_value",
        "monthly_sales",
        "category_sales",
        "insights",
    }

    missing_result_keys = required_result_keys.difference(
        result.keys()
    )

    if missing_result_keys:
        raise KeyError(
            "Sales intelligence result is missing keys: "
            + ", ".join(sorted(missing_result_keys))
        )

    # Retrieve the report data.
    monthly_sales = result["monthly_sales"].copy()
    category_sales = result["category_sales"].copy()
    insights = result["insights"]

    # Validate monthly report columns.
    required_monthly_columns = {
        "sales_month",
        "item_sales",
        "freight",
        "order_count",
        "sales_change_percent",
        "order_change_percent",
        "average_sales_per_order",
        "average_sales_per_order_change_percent",
    }

    missing_monthly_columns = required_monthly_columns.difference(
        monthly_sales.columns
    )

    if missing_monthly_columns:
        raise KeyError(
            "Monthly sales data is missing columns: "
            + ", ".join(sorted(missing_monthly_columns))
        )

    # Validate category report columns.
    required_category_columns = {
        "product_category",
        "item_sales",
        "freight",
        "item_records",
        "order_count",
        "sales_share_percent",
    }

    missing_category_columns = required_category_columns.difference(
        category_sales.columns
    )

    if missing_category_columns:
        raise KeyError(
            "Category sales data is missing columns: "
            + ", ".join(sorted(missing_category_columns))
        )

    # Build the monthly order activity report.
    monthly_activity, unmatched_items = (
        build_monthly_order_activity(data_dir)
    )

    # --------------------------------------------------
    # SAVE REPORT FILES
    # --------------------------------------------------

    monthly_sales.to_csv(
        reports_dir / "sales_monthly_delivered.csv",
        index=False,
    )

    monthly_activity.to_csv(
        reports_dir / "sales_monthly_order_activity.csv",
        index=False,
    )

    category_sales.to_csv(
        reports_dir / "sales_category_performance.csv",
        index=False,
    )

    # Save business insights as a readable text file.
    insights_path = reports_dir / "sales_insights.txt"

    with insights_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        file.write("SIGNALMIND SALES INSIGHTS\n")
        file.write("=" * 50 + "\n\n")

        for insight in insights:
            file.write(f"- {insight}\n")

    # --------------------------------------------------
    # DISPLAY REPORT
    # --------------------------------------------------

    print("=" * 60)
    print("SIGNALMIND SALES REPORT")
    print("=" * 60)

    print(
        "Total item records:",
        result["total_item_records"],
    )

    print(
        "Delivered orders:",
        result["total_delivered_orders"],
    )

    print(
        "Orders with item records:",
        result["orders_with_items"],
    )

    print(
        "Orders without item records:",
        result["orders_without_items"],
    )

    print(
        "Delivered item sales:",
        round(result["item_sales"], 2),
    )

    print(
        "Delivered freight total:",
        round(result["freight_total"], 2),
    )

    print(
        "Average item sales per order:",
        round(result["average_order_value"], 2),
    )

    print(
        "Average item value:",
        round(result["average_item_value"], 2),
    )

    print(
        "Unmatched item records:",
        unmatched_items,
    )

    # Display monthly sales metrics.
    print("\nMONTHLY DELIVERED SALES")

    display_columns = [
        "sales_month",
        "item_sales",
        "freight",
        "order_count",
        "sales_change_percent",
        "order_change_percent",
        "average_sales_per_order",
        "average_sales_per_order_change_percent",
    ]

    print(
        monthly_sales[display_columns]
        .round(2)
        .to_string(index=False)
    )

    # Display the top-selling categories.
    print("\nTOP 10 CATEGORIES BY ITEM SALES")

    print(
        category_sales.head(10)
        .round(2)
        .to_string(index=False)
    )

    # Display the insights generated by the intelligence engine.
    print("\nBUSINESS INSIGHTS")

    for insight in insights:
        print(f"- {insight}")

    # Confirm the generated output files.
    print("\nSAVED REPORTS")
    print("- reports/sales_monthly_delivered.csv")
    print("- reports/sales_monthly_order_activity.csv")
    print("- reports/sales_category_performance.csv")
    print("- reports/sales_insights.txt")


# --------------------------------------------------
# SCRIPT ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()