
from pathlib import Path

from intelligence.product_performance import build_product_performance


# 1. Locate the project reports folder
PROJECT = Path(__file__).resolve().parents[2]
REPORTS = PROJECT / "reports"

REPORTS.mkdir(exist_ok=True)


def main():
    # 2. Run the reusable intelligence module
    results = build_product_performance()

    products = results["product_performance"]
    categories = results["category_performance"]

    # 3. Save the detailed results
    products.to_csv(
        REPORTS / "product_performance.csv",
        index=False,
    )

    categories.to_csv(
        REPORTS / "category_performance.csv",
        index=False,
    )

    # 4. Print a concise summary
    print("=== SIGNALMIND PRODUCT PERFORMANCE REPORT ===")

    print("Delivered item rows:", results["delivered_item_rows"])
    print("Delivered orders:", results["delivered_order_count"])
    print(
        "Delivered item sales:",
        round(results["delivered_item_sales"], 2),
    )

    print(
        "Unmatched item-to-order records:",
        results["unmatched_order_items"],
    )

    print(
        "Delivered rows without matching product records:",
        results["unmatched_product_rows"],
    )

    print("\nTop 10 products by item sales:")

    print(
        products.head(10).round(2).to_string(index=False)
    )

    print("\nTop 10 categories by item sales:")

    print(
        categories.head(10).round(2).to_string(index=False)
    )

    print("\nSaved reports:")
    print("- reports/product_performance.csv")
    print("- reports/category_performance.csv")


if __name__ == "__main__":
    main()
