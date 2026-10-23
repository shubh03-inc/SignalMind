from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
SRC = PROJECT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intelligence.business_performance import build_business_performance

REPORTS = PROJECT / "reports"
REPORTS.mkdir(exist_ok=True)


def main():
    results = build_business_performance()

    # Export the report data.
    results["monthly_sales"].to_csv(
        REPORTS / "business_monthly_sales.csv",
        index=False,
    )

    results["category_sales"].to_csv(
        REPORTS / "business_category_sales.csv",
        index=False,
    )

    results["reconciliation_summary"].to_csv(
        REPORTS / "business_reconciliation_summary.csv",
        index=False,
    )

    results["status_summary"].to_csv(
        REPORTS / "business_reconciliation_status.csv",
        index=False,
    )

    # Print the business overview.
    print("=== SIGNALMIND BUSINESS PERFORMANCE REPORT ===")
    print(f"Total orders: {results['total_orders']:,}")
    print(f"Delivered orders: {results['total_delivered_orders']:,}")
    delivered_order_share = (
        results["total_delivered_orders"] / results["total_orders"] * 100
        if results["total_orders"] > 0
        else 0.0
    )

    print(f"Delivered order share: {delivered_order_share:.2f}%")
    print(f"Delivered item sales: {results['item_sales']:,.2f}")
    print(f"Delivered freight: {results['freight_total']:,.2f}")

    # Print data-quality metrics.
    print("\n=== DATA QUALITY ===")
    print(f"Orders without items: {results['orders_without_items']:,}")
    print(f"Orders without payments: {results['orders_without_payments']:,}")
    print(
        "Orders with large payment differences: "
        f"{results['orders_with_large_differences']:,}"
    )
    print(
        "Delivered orders with large payment differences: "
        f"{results['delivered_orders_with_large_differences']:,}"
    )
    print(
        "Unmatched order-item records: "
        f"{results['unmatched_order_items']:,}"
    )
    print(
        "Unmatched product records: "
        f"{results['unmatched_product_rows']:,}"
    )

    # Print the top-performing product.
    print("\n=== TOP-PERFORMING PRODUCT ===")
    top_product = results["top_product"]

    if top_product is not None:
        print(f"Product ID: {top_product['product_id']}")
        print(f"Item sales: {top_product['item_sales']:,.2f}")
        print(f"Item rows: {top_product['item_rows']}")
        print(f"Orders: {top_product['order_count']}")
    else:
        print("No product performance data available.")

    # Print the top-performing category.
    print("\n=== TOP-PERFORMING CATEGORY ===")
    top_category = results["top_category"]

    if top_category is not None:
        category_name = top_category.get(
            "category",
            top_category.get(
                "product_category",
                top_category.get("product_category_name", "Unknown"),
            ),
        )

        print(f"Category: {category_name}")
        print(f"Item sales: {top_category['item_sales']:,.2f}")
        print(
            "Sales contribution: "
            f"{top_category['sales_contribution_pct']:.2f}%"
        )
    else:
        print("No category performance data available.")

    # Print reconciliation results.
    print("\n=== RECONCILIATION SUMMARY ===")
    reconciliation = results["reconciliation_summary"]

    if not reconciliation.empty:
        print(reconciliation.round(2).to_string(index=False))
    else:
        print("No reconciliation data available.")

    # Print generated insights.
    print("\n=== BUSINESS INSIGHTS ===")
    for insight in results["insights"]:
        print(f"- {insight}")

    print("\n=== SAVED REPORTS ===")
    print("- reports/business_monthly_sales.csv")
    print("- reports/business_category_sales.csv")
    print("- reports/business_reconciliation_summary.csv")
    print("- reports/business_reconciliation_status.csv")


if __name__ == "__main__":
    main()
