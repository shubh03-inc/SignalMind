
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.reports.sales_report import build_monthly_order_activity


class TestMonthlyOrderActivity(unittest.TestCase):

    def setUp(self):
        self.orders = pd.DataFrame({
            "order_id": ["o1", "o2", "o3"],
            "order_status": ["delivered", "delivered", "canceled"],
            "order_purchase_timestamp": [
                "2018-01-10 10:00:00",
                "2018-01-20 12:00:00",
                "2018-02-05 09:00:00",
            ],
        })

        self.items = pd.DataFrame({
            "order_id": ["o1", "o1", "o2", "o3", "missing"],
        })

    def run_with_data(self, orders=None, items=None):
        orders = self.orders if orders is None else orders
        items = self.items if items is None else items

        def fake_read_csv(path, *args, **kwargs):
            if Path(path).name == "olist_orders_dataset.csv":
                return orders.copy()

            if Path(path).name == "olist_order_items_dataset.csv":
                return items.copy()

            raise AssertionError(f"Unexpected file: {path}")

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)

            # The production function checks that both files exist.
            (data_dir / "olist_orders_dataset.csv").touch()
            (data_dir / "olist_order_items_dataset.csv").touch()

            with patch(
                "src.reports.sales_report.pd.read_csv",
                side_effect=fake_read_csv,
            ):
                return build_monthly_order_activity(data_dir)

    def test_monthly_counts_are_correct(self):
        monthly, unmatched = self.run_with_data()

        january = monthly[
            (monthly["purchase_month"] == "2018-01")
            & (monthly["order_status"] == "delivered")
        ].iloc[0]

        february = monthly[
            (monthly["purchase_month"] == "2018-02")
            & (monthly["order_status"] == "canceled")
        ].iloc[0]

        self.assertEqual(january["item_rows"], 3)
        self.assertEqual(january["order_count"], 2)
        self.assertEqual(february["item_rows"], 1)
        self.assertEqual(february["order_count"], 1)
        self.assertEqual(unmatched, 1)

    def test_duplicate_order_ids_are_rejected(self):
        duplicate_orders = pd.concat(
            [self.orders, self.orders.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaisesRegex(ValueError, "Duplicate order IDs"):
            self.run_with_data(orders=duplicate_orders)

    def test_missing_order_id_is_rejected(self):
        invalid_orders = self.orders.copy()
        invalid_orders.loc[0, "order_id"] = None

        with self.assertRaisesRegex(ValueError, "missing order IDs"):
            self.run_with_data(orders=invalid_orders)

    def test_missing_item_order_id_is_rejected(self):
        invalid_items = self.items.copy()
        invalid_items.loc[0, "order_id"] = None

        with self.assertRaisesRegex(ValueError, "missing order IDs"):
            self.run_with_data(items=invalid_items)

    def test_invalid_timestamp_is_retained_in_missing_month_group(self):
        invalid_orders = self.orders.copy()
        invalid_orders.loc[0, "order_purchase_timestamp"] = "not-a-date"

        monthly, unmatched = self.run_with_data(orders=invalid_orders)

        missing_month = monthly[
            monthly["purchase_month"] == "NaT"
        ]

        self.assertEqual(missing_month["item_rows"].sum(), 3)
        self.assertEqual(unmatched, 1)


if __name__ == "__main__":
    unittest.main()
