from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd

# Make the SignalMind project root importable.
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from src.intelligence.sales_intelligence import build_sales_intelligence


# One cancelled order: the test dataset has no delivered orders.
orders = pd.DataFrame({
    "order_id": ["test-order-1"],
    "order_status": ["canceled"],
    "order_purchase_timestamp": ["2018-01-10 10:00:00"],
})

items = pd.DataFrame({
    "order_id": ["test-order-1"],
    "product_id": ["test-product-1"],
    "price": [100.0],
    "freight_value": [10.0],
})

products = pd.DataFrame({
    "product_id": ["test-product-1"],
    "product_category_name": ["test-category"],
})

# Replace CSV loading temporarily with our test data.
with patch(
    "src.intelligence.sales_intelligence.pd.read_csv",
    side_effect=[orders, items, products],
):
    result = build_sales_intelligence()

assert result["total_delivered_orders"] == 0
assert result["orders_with_items"] == 0
assert result["orders_without_items"] == 0
assert result["total_item_records"] == 0
assert result["item_sales"] == 0.0
assert result["freight_total"] == 0.0

print("PASS: Zero-delivered-orders test")
print("Delivered orders:", result["total_delivered_orders"])
print("Item sales:", result["item_sales"])
print("Freight:", result["freight_total"])