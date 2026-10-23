
"""General-purpose Sales Report analysis for uploaded datasets."""

import numpy as np
import pandas as pd


def analyze_uploaded_sales(
    dataframe,
    sales_column,
    date_column=None,
    category_column=None,
    order_id_column=None,
):
    """Analyze explicitly selected columns without modifying the input."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("The uploaded dataset has no rows.")

    if not sales_column or sales_column not in dataframe.columns:
        raise ValueError("Select a valid sales amount column.")

    for column in (date_column, category_column, order_id_column):
        if column is not None and column not in dataframe.columns:
            raise ValueError(f"Selected column does not exist: {column}")

    result_data = dataframe.copy()

    numeric_sales = pd.to_numeric(
        result_data[sales_column],
        errors="coerce",
    )

    finite_mask = pd.Series(
        np.isfinite(numeric_sales.fillna(0)),
        index=result_data.index,
    ) & numeric_sales.notna()

    invalid_sales_rows = int((~finite_mask).sum())

    # Analyze only rows with usable numeric sales values.
    analyzed = result_data.loc[finite_mask].copy()
    analyzed["_sales_value"] = numeric_sales.loc[finite_mask]

    if analyzed.empty:
        raise ValueError(
            "No valid numeric sales values were found in the selected column."
        )

    total_sales = float(analyzed["_sales_value"].sum())
    average_sales_per_row = float(analyzed["_sales_value"].mean())

    summary = {
        "rows_uploaded": int(len(result_data)),
        "rows_analyzed": int(len(analyzed)),
        "rows_excluded_invalid_sales": invalid_sales_rows,
        "total_sales": total_sales,
        "average_sales_per_row": average_sales_per_row,
        "sales_column": sales_column,
        "date_column": date_column,
        "category_column": category_column,
        "order_id_column": order_id_column,
    }

    if order_id_column:
        order_ids = analyzed[order_id_column]
        valid_order_mask = order_ids.notna() & (
            order_ids.astype(str).str.strip() != ""
        )
        valid_orders = analyzed.loc[valid_order_mask].copy()

        order_totals = valid_orders.groupby(
            order_id_column,
            dropna=True,
        )["_sales_value"].sum()

        summary["distinct_orders"] = int(order_totals.size)
        summary["average_sales_per_order"] = (
            float(order_totals.mean())
            if not order_totals.empty
            else None
        )
        summary["rows_without_order_id"] = int((~valid_order_mask).sum())
    else:
        summary["distinct_orders"] = None
        summary["average_sales_per_order"] = None
        summary["rows_without_order_id"] = None

    monthly_sales = pd.DataFrame(
        columns=["month", "sales", "records"]
    )

    if date_column:
        parsed_dates = pd.to_datetime(
            analyzed[date_column],
            errors="coerce",
        )
        dated = analyzed.loc[parsed_dates.notna()].copy()
        dated["_month"] = (
            parsed_dates.loc[parsed_dates.notna()]
            .dt.to_period("M")
            .astype(str)
        )

        if not dated.empty:
            monthly_sales = (
                dated.groupby("_month", as_index=False)
                .agg(
                    sales=("_sales_value", "sum"),
                    records=("_sales_value", "size"),
                )
                .rename(columns={"_month": "month"})
                .sort_values("month")
                .reset_index(drop=True)
            )

        summary["rows_with_invalid_dates"] = int(parsed_dates.isna().sum())
        summary["rows_with_valid_dates"] = int(parsed_dates.notna().sum())
    else:
        summary["rows_with_invalid_dates"] = None
        summary["rows_with_valid_dates"] = None

    category_sales = pd.DataFrame(
        columns=["category", "sales", "records"]
    )

    if category_column:
        categories = analyzed[category_column].copy()
        categories = categories.where(
            categories.notna(),
            "Unspecified",
        ).astype(str).str.strip()
        categories = categories.replace("", "Unspecified")

        categorized = analyzed.copy()
        categorized["_category"] = categories

        category_sales = (
            categorized.groupby("_category", as_index=False)
            .agg(
                sales=("_sales_value", "sum"),
                records=("_sales_value", "size"),
            )
            .rename(columns={"_category": "category"})
            .sort_values("sales", ascending=False)
            .reset_index(drop=True)
        )

    insights = [
        f"Analyzed {len(analyzed):,} rows with valid numeric sales values.",
        f"Total of selected sales values: {total_sales:,.2f}.",
    ]

    if invalid_sales_rows:
        insights.append(
            f"{invalid_sales_rows:,} rows were excluded because their "
            "sales values were missing, non-numeric, or non-finite."
        )

    if not monthly_sales.empty:
        highest_month = monthly_sales.loc[
            monthly_sales["sales"].idxmax(), "month"
        ]
        insights.append(
            f"Highest-total month in the analyzed data: {highest_month}."
        )

    if not category_sales.empty:
        top_category = category_sales.iloc[0]["category"]
        insights.append(f"Highest-total category: {top_category}.")

    return {
        "summary": summary,
        "monthly_sales": monthly_sales,
        "category_sales": category_sales,
        "insights": insights,
    }
