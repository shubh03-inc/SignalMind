
import pandas as pd
import pytest

from src.analysis.uploaded_sales import analyze_uploaded_sales


def test_calculates_sales_and_distinct_orders():
    data = pd.DataFrame(
        {
            "Order ID": ["A", "A", "B"],
            "Sales": [100, 50, 200],
        }
    )

    result = analyze_uploaded_sales(
        data,
        amount_column="Sales",
        order_id_column="Order ID",
    )

    summary = result["summary"]

    assert summary["total_sales_amount"] == 350
    assert summary["rows_used"] == 3
    assert summary["distinct_orders"] == 2
    assert summary["average_sales_per_order"] == 175


def test_groups_monthly_sales():
    data = pd.DataFrame(
        {
            "Date": ["2026-01-05", "2026-01-20", "2026-02-03"],
            "Sales": [100, 50, 200],
        }
    )

    result = analyze_uploaded_sales(
        data,
        amount_column="Sales",
        date_column="Date",
    )

    monthly = result["monthly_sales"]

    assert monthly["month"].tolist() == ["2026-01", "2026-02"]
    assert monthly["sales_amount"].tolist() == [150, 200]


def test_groups_category_sales_in_descending_order():
    data = pd.DataFrame(
        {
            "Category": ["Books", "Games", "Books"],
            "Sales": [100, 250, 50],
        }
    )

    result = analyze_uploaded_sales(
        data,
        amount_column="Sales",
        category_column="Category",
    )

    categories = result["category_sales"]

    assert categories["category"].tolist() == ["Games", "Books"]
    assert categories["sales_amount"].tolist() == [250, 150]


def test_reports_missing_and_invalid_amounts():
    data = pd.DataFrame(
        {
            "Sales": [100, None, "not numeric", 50],
        }
    )

    result = analyze_uploaded_sales(data, amount_column="Sales")
    summary = result["summary"]

    assert summary["total_sales_amount"] == 150
    assert summary["rows_in_dataset"] == 4
    assert summary["rows_used"] == 2
    assert summary["missing_amount_rows"] == 1
    assert summary["invalid_amount_rows"] == 1
    assert summary["rows_excluded_for_amount"] == 2


def test_rejects_missing_amount_column():
    data = pd.DataFrame({"Category": ["Books"]})

    with pytest.raises(ValueError, match="sales amount column"):
        analyze_uploaded_sales(data, amount_column="Sales")


def test_rejects_amount_column_without_numeric_values():
    data = pd.DataFrame({"Sales": ["unknown", "unavailable"]})

    with pytest.raises(ValueError, match="no usable numeric values"):
        analyze_uploaded_sales(data, amount_column="Sales")
