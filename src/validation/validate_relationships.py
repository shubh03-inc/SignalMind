
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"


def load(name, columns):
    return pd.read_csv(DATA / name, usecols=columns)


orders = load("olist_orders_dataset.csv", ["order_id", "customer_id"])
items = load(
    "olist_order_items_dataset.csv",
    ["order_id", "order_item_id", "product_id", "seller_id"],
)
customers = load("olist_customers_dataset.csv", ["customer_id"])
products = load("olist_products_dataset.csv", ["product_id", "product_category_name"])
sellers = load("olist_sellers_dataset.csv", ["seller_id"])
payments = load("olist_order_payments_dataset.csv", ["order_id"])
reviews = load("olist_order_reviews_dataset.csv", ["order_id"])
translations = load(
    "product_category_name_translation.csv",
    ["product_category_name"],
)

print("=== KEY UNIQUENESS ===")
checks = {
    "orders.order_id": orders["order_id"].is_unique,
    "customers.customer_id": customers["customer_id"].is_unique,
    "products.product_id": products["product_id"].is_unique,
    "sellers.seller_id": sellers["seller_id"].is_unique,
    "order item (order_id, order_item_id)": not items.duplicated(
        ["order_id", "order_item_id"]
    ).any(),
}

for name, passed in checks.items():
    print(f"{name}: {'PASS' if passed else 'FAIL'}")


def report_unmatched(label, source, column, target):
    unmatched = ~source[column].isin(target)
    print(f"{label}: {int(unmatched.sum())} unmatched rows")


print("\n=== RELATIONSHIP CHECKS ===")
report_unmatched("Items -> orders", items, "order_id", set(orders["order_id"]))
report_unmatched("Items -> products", items, "product_id", set(products["product_id"]))
report_unmatched("Items -> sellers", items, "seller_id", set(sellers["seller_id"]))
report_unmatched(
    "Orders -> customers",
    orders,
    "customer_id",
    set(customers["customer_id"]),
)
report_unmatched(
    "Payments -> orders",
    payments,
    "order_id",
    set(orders["order_id"]),
)
report_unmatched(
    "Reviews -> orders",
    reviews,
    "order_id",
    set(orders["order_id"]),
)

print("\n=== CATEGORY TRANSLATION ===")
known_categories = set(translations["product_category_name"].dropna())
categories = products["product_category_name"].dropna()
untranslated = ~categories.isin(known_categories)

print("Products with missing category:", int(
    products["product_category_name"].isna().sum()
))
print("Products with categories absent from translation table:", int(
    untranslated.sum()
))
print("\nValidation complete. Review results before joining datasets.")
