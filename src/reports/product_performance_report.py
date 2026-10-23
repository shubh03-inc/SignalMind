
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
SRC = PROJECT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intelligence.product_performance import build_product_performance

REPORTS = PROJECT / "reports"
REPORTS.mkdir(exist_ok=True)


def main():
    results = build_product_performance()

    products = results["product_performance"]
    categories = results["category_performance"]

    products.to_csv(
        REPORTS / "product_performance.csv",
        index=False,
    )
    categories.to_csv(
        REPORTS / "category_performance.csv",
        index=False,
    )

    print("=== SIGNALMIND PRODUCT PERFORMANCE REPORT ===")
    print("Delivered item rows:", results["delivered_item_rows"])
    print("Delivered orders:", results["delivered_order_count"])
    print("Delivered item sales:", round(results["delivered_item_sales"], 2))
    print("Unmatched item-to-order records:", results["unmatched_order_items"])
    print(
        "Delivered rows without matching product records:",
        results["unmatched_product_rows"],
    )

    print("\nTop 10 products by item sales:")
    print(products.head(10).round(2).to_string(index=False))

    print("\nTop 10 categories by item sales:")
    print(categories.head(10).round(2).to_string(index=False))

    print("\nSaved reports:")
    print("- reports/product_performance.csv")
    print("- reports/category_performance.csv")


if __name__ == "__main__":
    main()
