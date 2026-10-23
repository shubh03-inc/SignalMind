from src.intelligence.sales_intelligence import (
    build_sales_intelligence,
)
from src.intelligence.revenue_intelligence import (
    build_revenue_intelligence,
)
from src.intelligence.product_performance import (
    build_product_performance,
)


def build_business_performance():
    """
    Combine sales, revenue, and product indicators into
    an overall business performance assessment.

    This module orchestrates existing intelligence modules.
    It does not independently recalculate their core metrics.
    """

    # 1. Collect results from existing intelligence modules.
    sales = build_sales_intelligence()
    revenue = build_revenue_intelligence()
    products = build_product_performance()

    # 2. Extract sales metrics.
    total_delivered_orders = sales["total_delivered_orders"]
    item_sales = sales["item_sales"]
    freight_total = sales["freight_total"]

    # 3. Extract revenue reconciliation metrics.
    total_orders = revenue["total_orders"]

    orders_without_items = revenue["orders_without_items"]
    orders_without_payments = revenue["orders_without_payments"]

    orders_with_large_differences = (
        revenue["orders_with_large_differences"]
    )

    delivered_orders_with_large_differences = (
        revenue["delivered_orders_with_large_differences"]
    )

    # 4. Extract product performance metrics.
    product_table = products["product_performance"]
    category_table = products["category_performance"]

    unmatched_order_items = products["unmatched_order_items"]
    unmatched_product_rows = products["unmatched_product_rows"]

    top_product = (
        product_table.iloc[0].to_dict()
        if not product_table.empty
        else None
    )

    top_category = (
        category_table.iloc[0].to_dict()
        if not category_table.empty
        else None
    )

    # 5. Calculate overall business indicators.
    delivered_order_share = (
        total_delivered_orders / total_orders * 100
        if total_orders > 0
        else 0.0
    )

    large_difference_rate = (
        orders_with_large_differences / total_orders * 100
        if total_orders > 0
        else 0.0
    )

    # 6. Generate evidence-based business insights.
    insights = [
        (
            f"The dataset contains {total_orders:,} orders, "
            f"including {total_delivered_orders:,} delivered orders."
        ),
        (
            f"Delivered orders generated {item_sales:,.2f} "
            "in item sales, excluding freight."
        ),
        (
            f"Freight recorded against delivered orders totals "
            f"{freight_total:,.2f}."
        ),
        (
            f"{delivered_order_share:.2f}% of recorded orders "
            "have a delivered status."
        ),
    ]

    # Missing item records.
    if orders_without_items > 0:
        noun = (
            "order"
            if orders_without_items == 1
            else "orders"
        )

        insights.append(
            f"{orders_without_items:,} {noun} have no matching "
            "item records and require investigation."
        )

    # Missing payment records.
    if orders_without_payments > 0:
        noun = (
            "order"
            if orders_without_payments == 1
            else "orders"
        )

        insights.append(
            f"{orders_without_payments:,} {noun} have no matching "
            "payment records and require investigation."
        )

    # Payment differences.
    if orders_with_large_differences > 0:
        insights.append(
            f"{orders_with_large_differences:,} orders have payment "
            "differences exceeding the reconciliation threshold."
        )

    if delivered_orders_with_large_differences > 0:
        insights.append(
            f"{delivered_orders_with_large_differences:,} delivered "
            "orders have payment differences exceeding the threshold."
        )

    # Product performance insights.
    if top_product is not None:
        insights.append(
            f"Top product by item sales: "
            f"{top_product['product_id']} "
            f"({top_product['item_sales']:,.2f})."
        )

    if top_category is not None:
        category_name = top_category.get(
            "category",
            top_category.get("product_category", "Unknown"),
        )

        insights.append(
            f"Top product category by item sales: "
            f"{category_name} "
            f"({top_category['item_sales']:,.2f})."
        )

    # Product and order data-quality indicators.
    if unmatched_order_items > 0:
        insights.append(
            f"{unmatched_order_items:,} order-item records do not "
            "match known orders and require investigation."
        )

    if unmatched_product_rows > 0:
        insights.append(
            f"{unmatched_product_rows:,} delivered item rows do not "
            "match known product records."
        )

    # 7. Return structured results for the reporting layer.
    return {
        # Overall order metrics.
        "total_orders": total_orders,
        "total_delivered_orders": total_delivered_orders,
        "delivered_order_share_percent": delivered_order_share,

        # Sales metrics.
        "item_sales": item_sales,
        "freight_total": freight_total,
        "monthly_sales": sales["monthly_sales"],
        "category_sales": sales["category_sales"],

        # Revenue reconciliation metrics.
        "orders_without_items": orders_without_items,
        "orders_without_payments": orders_without_payments,
        "orders_with_large_differences": (
            orders_with_large_differences
        ),
        "large_difference_rate_percent": large_difference_rate,
        "delivered_orders_with_large_differences": (
            delivered_orders_with_large_differences
        ),
        "reconciliation_summary": revenue["reconciliation_summary"],
        "status_summary": revenue["status_summary"],

        # Product performance metrics.
        "top_product": top_product,
        "top_category": top_category,
        "unmatched_order_items": unmatched_order_items,
        "unmatched_product_rows": unmatched_product_rows,

        # Explainable business signals.
        "insights": insights,
    }


if __name__ == "__main__":
    result = build_business_performance()

    print("\n=== SIGNALMIND BUSINESS PERFORMANCE ===")

    print(f"Total orders: {result['total_orders']:,}")

    print(
        f"Delivered orders: "
        f"{result['total_delivered_orders']:,}"
    )

    print(
        f"Delivered order share: "
        f"{result['delivered_order_share_percent']:.2f}%"
    )

    print(
        f"Delivered item sales: "
        f"{result['item_sales']:,.2f}"
    )

    print(
        f"Delivered freight: "
        f"{result['freight_total']:,.2f}"
    )

    print(
        f"Orders without items: "
        f"{result['orders_without_items']:,}"
    )

    print(
        f"Orders without payments: "
        f"{result['orders_without_payments']:,}"
    )

    print(
        f"Orders with large payment differences: "
        f"{result['orders_with_large_differences']:,}"
    )

    print(
        f"Delivered orders with large payment differences: "
        f"{result['delivered_orders_with_large_differences']:,}"
    )

    print(
        f"Unmatched order-item records: "
        f"{result['unmatched_order_items']:,}"
    )

    print(
        f"Unmatched product records: "
        f"{result['unmatched_product_rows']:,}"
    )

    print("\n=== TOP-PERFORMING PRODUCT ===")

    if result["top_product"] is not None:
        print(result["top_product"])
    else:
        print("No product performance data available.")

    print("\n=== TOP-PERFORMING CATEGORY ===")

    if result["top_category"] is not None:
        print(result["top_category"])
    else:
        print("No category performance data available.")

    print("\n=== RECONCILIATION SUMMARY ===")

    print(
        result["reconciliation_summary"].to_string(
            index=False
        )
    )

    print("\n=== BUSINESS INSIGHTS ===")

    for insight in result["insights"]:
        print(f"- {insight}")
