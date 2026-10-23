
import pandas as pd
import pytest

from src.analysis.upload_sales import analyze_uploaded_sales


def test_sales_totals_and_order_average():
    data = pd.DataFrame({
        "Order": ["A", "A", "B"],
        "Amount": [10, 15, 25],
    })

    result = analyze_uploaded_sales(
        data,
        sales_column="Amount",
        order_id_column="Order",
    )

    summary = result["summary"]

    assert summary["total_sales"] == 50
    assert summary["distinct_orders"] == 2
    assert summary["average_sales_per_order"] == 25
    assert summary["rows_analyzed"] == 3


def test_invalid_sales_values_are_excluded():
    data = pd.DataFrame({
        "Amount": [10, "bad", None, 20],
    })

    result = analyze_uploaded_sales(data, sales_column="Amount")

    assert result["summary"]["total_sales"] == 30
    assert result["summary"]["rows_analyzed"] == 2
    assert result["summary"]["rows_excluded_invalid_sales"] == 2


def test_monthly_sales_are_grouped_correctly():
    data = pd.DataFrame({
        "Date": ["2026-01-03", "2026-01-19", "2026-02-01"],
        "Amount": [10, 20, 30],
    })

    result = analyze_uploaded_sales(
        data,
        sales_column="Amount",
        date_column="Date",
    )

    monthly = result["monthly_sales"]

    assert monthly["month"].tolist() == ["2026-01", "2026-02"]
    assert monthly["sales"].tolist() == [30, 30]


def test_category_sales_are_sorted_by_total():
    data = pd.DataFrame({
        "Category": ["A", "B", "A"],
        "Amount": [10, 40, 20],
    })

    result = analyze_uploaded_sales(
        data,
        sales_column="Amount",
        category_column="Category",
    )

    categories = result["category_sales"]

    assert categories.iloc[0]["category"] == "B"
    assert categories.iloc[0]["sales"] == 40


def test_empty_dataset_is_rejected():
    data = pd.DataFrame({"Amount": []})

    with pytest.raises(ValueError, match="no rows"):
        analyze_uploaded_sales(data, sales_column="Amount")


def test_missing_sales_column_is_rejected():
    data = pd.DataFrame({"Revenue": [10, 20]})

    with pytest.raises(ValueError, match="Select a valid sales amount"):
        analyze_uploaded_sales(data, sales_column="Amount")


def test_negative_sales_are_preserved():
    data = pd.DataFrame({"Amount": [100, -20]})

    result = analyze_uploaded_sales(data, sales_column="Amount")

    assert result["summary"]["total_sales"] == 80
