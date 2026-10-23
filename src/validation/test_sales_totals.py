from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from src.intelligence.sales_intelligence import build_sales_intelligence

result = build_sales_intelligence()

monthly = result["monthly_sales"]
categories = result["category_sales"]

total_sales = result["item_sales"]

monthly_total = monthly["item_sales"].sum()
category_total = categories["item_sales"].sum()

print("Overall item sales:", round(total_sales, 2))
print("Monthly item sales:", round(monthly_total, 2))
print("Category item sales:", round(category_total, 2))

assert abs(monthly_total - total_sales) < 0.02, (
    "Monthly sales do not match overall item sales"
)

assert abs(category_total - total_sales) < 0.02, (
    "Category sales do not match overall item sales"
)

print("PASS: Monthly and category totals match overall sales")