
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data"


def build_product_performance():
    """Calculate product and category performance for delivered orders."""

    items = pd.read_csv(
        DATA / "olist_order_items_dataset.csv",
        usecols=["order_id", "product_id", "price"],
    )

    orders = pd.read_csv(
        DATA / "olist_orders_dataset.csv",
        usecols=["order_id", "order_status"],
    )

    products = pd.read_csv(
        DATA / "olist_products_dataset.csv",
        usecols=["product_id", "product_category_name"],
    )

    translations = pd.read_csv(
        DATA / "product_category_name_translation.csv"
    )

    # Validate the numeric data and join keys.
    items["price"] = pd.to_numeric(
        items["price"],
        errors="coerce",
    )

    if items["price"].isna().any():
        raise ValueError("Item prices contain missing or invalid values.")

    if (items["price"] < 0).any():
        raise ValueError("Negative item prices require investigation.")

    if orders["order_id"].duplicated().any():
        raise ValueError("Duplicate order IDs found.")

    if products["product_id"].duplicated().any():
        raise ValueError("Duplicate product IDs found.")

    if translations["product_category_name"].duplicated().any():
        raise ValueError("Duplicate category translation keys found.")

    # Attach order status to each item.
    item_data = items.merge(
        orders,
        on="order_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched_order_items = int(
        item_data["_merge"].eq("left_only").sum()
    )

    item_data = item_data.drop(columns="_merge")

    if len(item_data) != len(items):
        raise ValueError("Joining orders changed the item row count.")

    # Product performance uses delivered orders only.
    delivered = item_data[
        item_data["order_status"].eq("delivered")
    ].copy()

    delivered = delivered.merge(
        products,
        on="product_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched_product_rows = int(
        delivered["_merge"].eq("left_only").sum()
    )

    delivered = delivered.drop(columns="_merge")

    # Translate category names into English where available.
    delivered = delivered.merge(
        translations,
        on="product_category_name",
        how="left",
        validate="many_to_one",
    )

    delivered["category"] = (
        delivered["product_category_name_english"]
        .fillna(delivered["product_category_name"])
        .fillna("Unknown category")
    )

    # Aggregate results for individual products.
    product_performance = (
        delivered.groupby("product_id", dropna=False)
        .agg(
            item_sales=("price", "sum"),
            item_rows=("order_id", "size"),
            order_count=("order_id", "nunique"),
            average_item_price=("price", "mean"),
        )
        .reset_index()
        .sort_values("item_sales", ascending=False)
    )

    # Aggregate results for product categories.
    category_performance = (
        delivered.groupby("category", dropna=False)
        .agg(
            item_sales=("price", "sum"),
            item_rows=("order_id", "size"),
            order_count=("order_id", "nunique"),
            average_item_price=("price", "mean"),
            distinct_products=("product_id", "nunique"),
        )
        .reset_index()
        .sort_values("item_sales", ascending=False)
    )

    total_sales = delivered["price"].sum()

    if total_sales > 0:
        category_performance["sales_contribution_pct"] = (
            category_performance["item_sales"]
            / total_sales
            * 100
        )
    else:
        category_performance["sales_contribution_pct"] = 0.0

    return {
        "product_performance": product_performance,
        "category_performance": category_performance,
        "delivered_item_sales": float(total_sales),
        "delivered_item_rows": len(delivered),
        "delivered_order_count": delivered["order_id"].nunique(),
        "unmatched_order_items": unmatched_order_items,
        "unmatched_product_rows": unmatched_product_rows,
    }
