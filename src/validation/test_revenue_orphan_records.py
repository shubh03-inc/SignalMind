import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.intelligence import revenue_intelligence


class TestRevenueOrphanRecords(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)

        self.orders_path = self.data_dir / "olist_orders_dataset.csv"
        self.items_path = self.data_dir / "olist_order_items_dataset.csv"
        self.payments_path = self.data_dir / "olist_order_payments_dataset.csv"

        self.orders = pd.DataFrame(
            [
                {
                    "order_id": "order_1",
                    "order_status": "delivered",
                }
            ]
        )

        self.items = pd.DataFrame(
            [
                {
                    "order_id": "order_1",
                    "price": 100.0,
                    "freight_value": 10.0,
                }
            ]
        )

        self.payments = pd.DataFrame(
            [
                {
                    "order_id": "order_1",
                    "payment_value": 110.0,
                }
            ]
        )

        self._write_test_data(
            self.orders,
            self.items,
            self.payments,
        )

        self.data_patch = patch.object(
            revenue_intelligence,
            "DATA",
            self.data_dir,
        )
        self.data_patch.start()

        self.addCleanup(self.data_patch.stop)
        self.addCleanup(self.temp_dir.cleanup)

    def _write_test_data(self, orders, items, payments):
        """Write test DataFrames to temporary CSV files."""
        orders.to_csv(self.orders_path, index=False)
        items.to_csv(self.items_path, index=False)
        payments.to_csv(self.payments_path, index=False)

    def test_rejects_item_with_unknown_order_id(self):
        """Reject item records referencing nonexistent orders."""
        items = pd.DataFrame(
            [
                {
                    "order_id": "unknown_order",
                    "price": 100.0,
                    "freight_value": 10.0,
                }
            ]
        )

        self._write_test_data(
            self.orders,
            items,
            self.payments,
        )

        with self.assertRaisesRegex(
            ValueError,
            "unknown order IDs",
        ) as context:
            revenue_intelligence.build_revenue_intelligence()

        self.assertIn(
            "unknown_order",
            str(context.exception),
        )

    def test_rejects_payment_with_unknown_order_id(self):
        """Reject payment records referencing nonexistent orders."""
        payments = pd.DataFrame(
            [
                {
                    "order_id": "unknown_order",
                    "payment_value": 110.0,
                }
            ]
        )

        self._write_test_data(
            self.orders,
            self.items,
            payments,
        )

        with self.assertRaisesRegex(
            ValueError,
            "unknown order IDs",
        ) as context:
            revenue_intelligence.build_revenue_intelligence()

        self.assertIn(
            "unknown_order",
            str(context.exception),
        )

    def test_valid_records_reconcile_without_difference(self):
        """Valid records with matching amounts should reconcile."""
        self._write_test_data(
            self.orders,
            self.items,
            self.payments,
        )

        result = revenue_intelligence.build_revenue_intelligence()

        self.assertEqual(result["total_orders"], 1)
        self.assertEqual(result["orders_with_items"], 1)
        self.assertEqual(result["orders_with_payments"], 1)
        self.assertEqual(result["orders_with_large_differences"], 0)

        reconciliation = result["reconciliation"]

        self.assertAlmostEqual(
            reconciliation.iloc[0]["item_sales"],
            100.0,
        )
        self.assertAlmostEqual(
            reconciliation.iloc[0]["freight"],
            10.0,
        )
        self.assertAlmostEqual(
            reconciliation.iloc[0]["payment_total"],
            110.0,
        )
        self.assertEqual(
            reconciliation.iloc[0]["reconciliation_status"],
            "RECONCILED",
        )

    def test_missing_items_are_flagged(self):
        """Orders with payments but no item records must be flagged."""
        empty_items = pd.DataFrame(
            columns=[
                "order_id",
                "price",
                "freight_value",
            ]
        )

        self._write_test_data(
            self.orders,
            empty_items,
            self.payments,
        )

        result = revenue_intelligence.build_revenue_intelligence()

        self.assertEqual(result["orders_without_items"], 1)
        self.assertEqual(
            len(result["missing_items_with_payments"]),
            1,
        )

        reconciliation = result["reconciliation"]

        self.assertTrue(
            reconciliation.iloc[0]["missing_items"]
        )
        self.assertIn(
            "No item records",
            reconciliation.iloc[0]["issue"],
        )
        self.assertEqual(
            reconciliation.iloc[0]["reconciliation_status"],
            "MISSING_ITEMS",
        )

    def test_rejects_duplicate_order_ids(self):
        """Duplicate order IDs in the orders table must be rejected."""
        duplicate_orders = pd.DataFrame(
            [
                {
                    "order_id": "order_1",
                    "order_status": "delivered",
                },
                {
                    "order_id": "order_1",
                    "order_status": "shipped",
                },
            ]
        )

        self._write_test_data(
            duplicate_orders,
            self.items,
            self.payments,
        )

        with self.assertRaisesRegex(
            ValueError,
            "duplicate order IDs",
        ):
            revenue_intelligence.build_revenue_intelligence()

    def test_rejects_negative_item_price(self):
        """Negative item prices must be rejected."""
        negative_items = pd.DataFrame(
            [
                {
                    "order_id": "order_1",
                    "price": -10.0,
                    "freight_value": 2.0,
                }
            ]
        )

        self._write_test_data(
            self.orders,
            negative_items,
            self.payments,
        )

        with self.assertRaisesRegex(
            ValueError,
            "negative values",
        ):
            revenue_intelligence.build_revenue_intelligence()

    def test_rejects_non_numeric_item_price(self):
        """Non-numeric item prices must be rejected."""
        invalid_items = pd.DataFrame(
            [
                {
                    "order_id": "order_1",
                    "price": "invalid",
                    "freight_value": 10.0,
                }
            ]
        )

        self._write_test_data(
            self.orders,
            invalid_items,
            self.payments,
        )

        with self.assertRaisesRegex(
            ValueError,
            "invalid or non-finite values",
        ):
            revenue_intelligence.build_revenue_intelligence()

    
    
    def test_rejects_infinite_item_price(self):
        """Infinite item prices must be rejected."""
        infinite_items = pd.DataFrame(
            [
                {
                    "order_id": "order_1",
                    "price": float("inf"),
                    "freight_value": 10.0,
                }
            ]
        )

        self._write_test_data(
            self.orders,
            infinite_items,
            self.payments,
        )

        with self.assertRaisesRegex(
            ValueError,
            "invalid or non-finite values",
        ):
            revenue_intelligence.build_revenue_intelligence()


    def test_rejects_missing_item_price(self):
        """Missing item prices must be rejected."""
        missing_price_items = pd.DataFrame(
            [
                {
                    "order_id": "order_1",
                    "price": float("nan"),
                    "freight_value": 10.0,
                }
            ]
        )

        self._write_test_data(
            self.orders,
            missing_price_items,
            self.payments,
        )

        with self.assertRaisesRegex(
            ValueError,
            "invalid or non-finite values",
        ):
            revenue_intelligence.build_revenue_intelligence()

    
    def test_rejects_negative_payment_value(self):
        """Negative payment values must be rejected."""
        negative_payments = pd.DataFrame(
            [
                {
                    "order_id": "order_1",
                    "payment_value": -110.0,
                }
            ]
        )

        self._write_test_data(
            self.orders,
            self.items,
            negative_payments,
        )

        with self.assertRaisesRegex(
            ValueError,
            "negative values",
        ):
            revenue_intelligence.build_revenue_intelligence()

    
    def test_missing_items_take_priority_over_amount_mismatch(self):
        """Missing items take status priority while retaining mismatch flags."""
        empty_items = pd.DataFrame(
            columns=["order_id", "price", "freight_value"]
        )

        payments = pd.DataFrame(
            [
                {
                    "order_id": "order_1",
                    "payment_value": 110.0,
                }
            ]
        )

        self._write_test_data(
            self.orders,
            empty_items,
            payments,
        )

        result = revenue_intelligence.build_revenue_intelligence()
        row = result["reconciliation"].iloc[0]

        self.assertTrue(row["missing_items"])
        self.assertTrue(row["large_difference"])
        self.assertEqual(
            row["reconciliation_status"],
            "MISSING_ITEMS",
        )
        self.assertIn("No item records", row["issue"])
        self.assertIn(
            "Payment difference exceeds threshold",
            row["issue"],
        )



if __name__ == "__main__":
    unittest.main()