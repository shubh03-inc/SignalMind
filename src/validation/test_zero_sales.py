from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from src.intelligence.sales_intelligence import build_sales_intelligence


orders = pd.DataFrame({
    "order_id": ["test-order-1"],
    "order_status": ["delivered"],
    "order_purchase_timestamp": ["2018-01-10 10:00:00"],
})

items = pd.DataFrame({
    "order_id": ["test-order-1"],
    "product_id": ["test-product-1"],
    "price": [0.0],
    "freight_value": [0.0],
})

products = pd.DataFrame({
    "product_id": ["test-product-1"],
    "product_category_name": ["test-category"],
})

with patch(
    "src.intelligence.sales_intelligence.pd.read_csv",
    side_effect=[orders, items, products],
):
    result = build_sales_intelligence()

assert result["total_delivered_orders"] == 1
assert result["total_item_records"] == 1
assert result["item_sales"] == 0.0
assert result["freight_total"] == 0.0

categories = result["category_sales"]

assert categories["sales_share_percent"].eq(0.0).all()

print("PASS: Zero-sales test")
print("Delivered orders:", result["total_delivered_orders"])
print("Item sales:", result["item_sales"])
print("Category sales shares:", categories["sales_share_percent"].tolist())