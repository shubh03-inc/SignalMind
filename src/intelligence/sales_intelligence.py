
from pathlib import Path
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"


def build_sales_intelligence():
    """
    Analyze sales trends, order volume, and product categories.

    Primary sales metrics use delivered orders and item prices.
    Freight is reported separately.

    Sales are grouped by order purchase month, not delivery month.
    These figures are not necessarily accounting-recognized revenue.
    """

    # 1. Load datasets
    orders = pd.read_csv(
        DATA / "olist_orders_dataset.csv",
        usecols=[
            "order_id",
            "order_status",
            "order_purchase_timestamp",
        ],
    )

    items = pd.read_csv(
        DATA / "olist_order_items_dataset.csv",
        usecols=[
            "order_id",
            "product_id",
            "price",
            "freight_value",
        ],
    )

    products = pd.read_csv(
        DATA / "olist_products_dataset.csv",
        usecols=[
            "product_id",
            "product_category_name",
        ],
    )

    # 2. Validate essential fields
    if orders["order_id"].isna().any():
        raise ValueError("Orders contain missing order IDs.")

    if orders["order_id"].duplicated().any():
        raise ValueError("Duplicate order IDs found in orders.")

    if items["order_id"].isna().any():
        raise ValueError("Order items contain missing order IDs.")

    if items["product_id"].isna().any():
        raise ValueError("Order items contain missing product IDs.")

    if products["product_id"].isna().any():
        raise ValueError("Products contain missing product IDs.")

    if products["product_id"].duplicated().any():
        raise ValueError("Duplicate product IDs found in products.")

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce",
    )

    if orders["order_purchase_timestamp"].isna().any():
        raise ValueError(
            "Invalid or missing order purchase timestamps found."
        )

    for column in ["price", "freight_value"]:
        items[column] = pd.to_numeric(
            items[column],
            errors="coerce",
        )

    if items[["price", "freight_value"]].isna().any().any():
        raise ValueError(
            "Invalid or missing item prices or freight values found."
        )

    if (items[["price", "freight_value"]] < 0).any().any():
        raise ValueError(
            "Negative prices or freight values require investigation."
        )

    # 3. Validate order relationships
    unknown_item_orders = (
        set(items["order_id"]) - set(orders["order_id"])
    )

    if unknown_item_orders:
        raise ValueError(
            f"Found {len(unknown_item_orders)} item records "
            "whose order IDs do not exist in the orders dataset."
        )

    # 4. Select delivered orders
    delivered_orders = orders.loc[
        orders["order_status"].eq("delivered")
    ].copy()

    delivered_ids = set(delivered_orders["order_id"])

    delivered_items = items.loc[
        items["order_id"].isin(delivered_ids)
    ].copy()

    # 5. Overall sales metrics
    item_sales = delivered_items["price"].sum()
    freight_total = delivered_items["freight_value"].sum()

    total_delivered_orders = delivered_orders["order_id"].nunique()
    orders_with_items = delivered_items["order_id"].nunique()

    orders_without_items = (
        total_delivered_orders - orders_with_items
    )

    average_order_value = (
        item_sales / orders_with_items
        if orders_with_items > 0
        else 0.0
    )

    average_item_value = (
        item_sales / len(delivered_items)
        if not delivered_items.empty
        else 0.0
    )

    total_item_records = len(delivered_items)

    # 6. Monthly sales trends
    delivered_items = delivered_items.merge(
        delivered_orders[
            ["order_id", "order_purchase_timestamp"]
        ],
        on="order_id",
        how="left",
        validate="many_to_one",
    )

    delivered_items["sales_month"] = (
        delivered_items["order_purchase_timestamp"].dt.to_period("M")
    )

    monthly_sales = (
        delivered_items.groupby("sales_month")
        .agg(
            item_sales=("price", "sum"),
            freight=("freight_value", "sum"),
            item_records=("order_id", "size"),
            order_count=("order_id", "nunique"),
        )
        .sort_index()
    )

    if not monthly_sales.empty:
        complete_months = pd.period_range(
            start=monthly_sales.index.min(),
            end=monthly_sales.index.max(),
            freq="M",
        )

        monthly_sales = monthly_sales.reindex(complete_months)

        metric_columns = [
            "item_sales",
            "freight",
            "item_records",
            "order_count",
        ]

        monthly_sales[metric_columns] = (
            monthly_sales[metric_columns].fillna(0)
        )

    monthly_sales.index.name = "sales_month"
    monthly_sales = monthly_sales.reset_index()

    monthly_sales["sales_month"] = (
        monthly_sales["sales_month"].astype(str)
    )

    # 7. Month-over-month sales changes
    monthly_sales["previous_month_sales"] = (
        monthly_sales["item_sales"].shift(1)
    )

    monthly_sales["sales_change"] = (
        monthly_sales["item_sales"]
        - monthly_sales["previous_month_sales"]
    )

    previous_sales = monthly_sales[
        "previous_month_sales"
    ].where(
        monthly_sales["previous_month_sales"] > 0
    )

    monthly_sales["sales_change_percent"] = (
        monthly_sales["sales_change"]
        .div(previous_sales)
        .mul(100)
    )

    # Percentage change is unavailable when the current month
    # has no recorded orders.
    no_current_orders = monthly_sales["order_count"] == 0

    monthly_sales.loc[
        no_current_orders,
        "sales_change_percent",
    ] = float("nan")

    # 8. Month-over-month order-count changes
    monthly_sales["previous_month_orders"] = (
        monthly_sales["order_count"].shift(1)
    )

    previous_orders = monthly_sales[
        "previous_month_orders"
    ].where(
        monthly_sales["previous_month_orders"] > 0
    )

    monthly_sales["order_change_percent"] = (
        monthly_sales["order_count"]
        .sub(monthly_sales["previous_month_orders"])
        .div(previous_orders)
        .mul(100)
    )

    # Percentage change is unavailable when the current month
    # has no recorded orders.
    no_current_orders = monthly_sales["order_count"] == 0

    monthly_sales.loc[
        no_current_orders,
        "order_change_percent",
    ] = float("nan")

    # 9. Average item sales per order
    monthly_sales["average_sales_per_order"] = (
        monthly_sales["item_sales"]
        .div(
            monthly_sales["order_count"].where(
                monthly_sales["order_count"] > 0
            )
        )
    )

    monthly_sales["previous_average_sales_per_order"] = (
        monthly_sales["average_sales_per_order"].shift(1)
    )

    previous_average = (
        monthly_sales["previous_average_sales_per_order"].where(
            monthly_sales["previous_average_sales_per_order"] > 0
        )
    )

    monthly_sales["average_sales_per_order_change_percent"] = (
        monthly_sales["average_sales_per_order"]
        .sub(monthly_sales["previous_average_sales_per_order"])
        .div(previous_average)
        .mul(100)
    )

    # 10. Sales by product category
    category_items = delivered_items.merge(
        products,
        on="product_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    category_items["product_category_name"] = (
        category_items["product_category_name"].fillna("Unknown")
    )

    category_items.loc[
        category_items["_merge"].eq("left_only"),
        "product_category_name",
    ] = "Unmatched product"

    category_sales = (
        category_items.groupby("product_category_name")
        .agg(
            item_sales=("price", "sum"),
            freight=("freight_value", "sum"),
            item_records=("order_id", "size"),
            order_count=("order_id", "nunique"),
        )
        .reset_index()
        .rename(
            columns={
                "product_category_name": "product_category"
            }
        )
        .sort_values(
            "item_sales",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    if item_sales > 0:
        category_sales["sales_share_percent"] = (
            category_sales["item_sales"] / item_sales * 100
        )
    else:
        category_sales["sales_share_percent"] = 0.0

    # 11. Highest and lowest sales months
    if not monthly_sales.empty:
        highest_sales_month = monthly_sales.loc[
            monthly_sales["item_sales"].idxmax()
        ].to_dict()

        lowest_sales_month = monthly_sales.loc[
            monthly_sales["item_sales"].idxmin()
        ].to_dict()
    else:
        highest_sales_month = None
        lowest_sales_month = None

    # 12. Generate explainable business signals
    insights = [
        f"Delivered orders generated {item_sales:,.2f} in item sales, excluding freight.",
        f"The dataset contains {total_delivered_orders:,} delivered orders.",
    ]

    if orders_without_items > 0:
        insights.append(
            f"{orders_without_items:,} delivered orders have no matching "
            "item records and require investigation."
        )
    


        # Report delivered items whose product category is missing.
    unknown_category_items = category_items[
        category_items["product_category_name"].eq("Unknown")
    ]

    unknown_category_count = len(unknown_category_items)

    if unknown_category_count > 0:
        unknown_category_share = (
            unknown_category_count / len(delivered_items) * 100
            if len(delivered_items) > 0
            else 0.0
        )

        insights.append(
            f"{unknown_category_count:,} delivered item records "
            f"({unknown_category_share:.2f}%) have missing product "
            "categories. Category-level conclusions may be incomplete."
        )


    if not category_sales.empty:
        top_category = category_sales.iloc[0]
        insights.append(
            f"The highest-sales category is "
            f"'{top_category['product_category']}', with "
            f"{top_category['item_sales']:,.2f} in item sales."
        )

    if highest_sales_month is not None:
        insights.append(
            f"The highest-sales month in the available data is "
            f"{highest_sales_month['sales_month']}."
        )

    if len(monthly_sales) >= 2:
        latest = monthly_sales.iloc[-1]

        sales_change = latest["sales_change_percent"]
        order_change = latest["order_change_percent"]
        average_change = latest[
            "average_sales_per_order_change_percent"
        ]

        if pd.notna(sales_change):
            if sales_change > 0:
                direction = "increased"
            elif sales_change < 0:
                direction = "decreased"
            else:
                direction = "remained unchanged"

            insights.append(
                f"Item sales {direction} by "
                f"{abs(sales_change):.2f}% compared with "
                "the previous calendar month."
            )

        if (
            pd.notna(sales_change)
            and pd.notna(order_change)
            and pd.notna(average_change)
        ):
            if sales_change < 0 and order_change > 0:
                insights.append(
                    "Sales fell while order volume increased. "
                    f"Item sales changed by {sales_change:.2f}%, "
                    f"orders changed by {order_change:.2f}%, "
                    f"and average item sales per order changed "
                    f"by {average_change:.2f}%. "
                    "Investigate product mix, pricing, and discounts "
                    "before attributing a cause."
                )

            elif sales_change > 0 and order_change < 0:
                insights.append(
                    "Sales increased while order volume declined. "
                    f"Item sales changed by {sales_change:.2f}%, "
                    f"orders changed by {order_change:.2f}%, "
                    f"and average item sales per order changed "
                    f"by {average_change:.2f}%. "
                    "Investigate product mix and order values "
                    "before attributing a cause."
                )

            elif sales_change < 0 and order_change < 0:
                insights.append(
                    "Both item sales and order volume declined. "
                    f"Item sales changed by {sales_change:.2f}% "
                    f"and orders changed by {order_change:.2f}%. "
                    "Investigate demand, availability, and sales "
                    "channels."
                )

            elif sales_change > 0 and order_change > 0:
                insights.append(
                    "Both item sales and order volume increased. "
                    f"Item sales changed by {sales_change:.2f}% "
                    f"and orders changed by {order_change:.2f}%. "
                    "Investigate which products contributed "
                    "to the growth."
                )

    # Flag months whose previous-month order baseline is small.
     
    # Flag months whose previous-month sales baseline is small.
    low_sales_baseline_months = monthly_sales.loc[
        monthly_sales["previous_month_sales"].between(
            0.01, 999.99
        )
        & monthly_sales["order_count"].gt(0),
        ["sales_month", "previous_month_sales"],
    ]

    for _, month in low_sales_baseline_months.iterrows():
        previous_sales = float(month["previous_month_sales"])

        insights.append(
            f"The sales comparison for {month['sales_month']} "
            f"starts from only {previous_sales:,.2f} in "
            "previous-month item sales. Interpret the percentage "
            "change cautiously."
        )
    
    
    small_baseline_months = monthly_sales.loc[
        monthly_sales["previous_month_orders"].between(1, 99)
        & monthly_sales["order_count"].gt(0),
        ["sales_month", "previous_month_orders"],
    ]

    for _, month in small_baseline_months.iterrows():
        previous_orders = int(month["previous_month_orders"])
        noun = "order" if previous_orders == 1 else "orders"

        insights.append(
            f"The order comparison for {month['sales_month']} "
            f"starts from only {previous_orders} previous-month "
            f"{noun}. Interpret the percentage change cautiously."
        )

    # 13. Return results
    return {
        "total_delivered_orders": total_delivered_orders,
        "orders_with_items": orders_with_items,
        "orders_without_items": orders_without_items,
        "total_item_records": total_item_records,
        "item_sales": float(item_sales),
        "freight_total": float(freight_total),
        "average_order_value": float(average_order_value),
        "average_item_value": float(average_item_value),
        "monthly_sales": monthly_sales,
        "category_sales": category_sales,
        "highest_sales_month": highest_sales_month,
        "lowest_sales_month": lowest_sales_month,
        "insights": insights,
    }

if __name__ == "__main__":
    result = build_sales_intelligence()

    print("\n=== SALES INTELLIGENCE ===")
    print(f"Delivered orders: {result['total_delivered_orders']:,}")
    print(f"Orders with items: {result['orders_with_items']:,}")
    print(f"Orders without items: {result['orders_without_items']:,}")
    print(f"Item sales: {result['item_sales']:,.2f}")
    print(f"Freight total: {result['freight_total']:,.2f}")
    print(f"Average order value: {result['average_order_value']:,.2f}")
    print(f"Average item value: {result['average_item_value']:,.2f}")

    print("\n=== MONTHLY SALES ===")
    print(result["monthly_sales"].to_string(index=False))

    print("\n=== TOP PRODUCT CATEGORIES ===")
    print(result["category_sales"].head(10).to_string(index=False))

    print("\n=== BUSINESS INSIGHTS ===")
    for insight in result["insights"]:
        print(f"- {insight}")