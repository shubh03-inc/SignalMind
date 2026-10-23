import unittest
from unittest.mock import patch

import pandas as pd

from src.intelligence.product_performance import build_product_performance


class TestProductPerformance(unittest.TestCase):

    def setUp(self):
        # Three delivered item records and one non-delivered record.
        self.items = pd.DataFrame({
            "order_id": ["o1", "o1", "o2", "o3"],
            "product_id": ["p1", "p2", "p1", "p2"],
            "price": [10.0, 20.0, 30.0, 100.0],
        })

        self.orders = pd.DataFrame({
            "order_id": ["o1", "o2", "o3"],
            "order_status": ["delivered", "delivered", "canceled"],
        })

        self.products = pd.DataFrame({
            "product_id": ["p1", "p2"],
            "product_category_name": ["cat_a", "cat_b"],
        })

        self.translations = pd.DataFrame({
            "product_category_name": ["cat_a", "cat_b"],
            "product_category_name_english": [
                "category_a",
                "category_b",
            ],
        })

    def run_with_data(self):
        datasets = {
            "olist_order_items_dataset.csv": self.items,
            "olist_orders_dataset.csv": self.orders,
            "olist_products_dataset.csv": self.products,
            "product_category_name_translation.csv": self.translations,
        }

        def fake_read_csv(path, *args, **kwargs):
            filename = str(path).replace("\\", "/").split("/")[-1]

            if filename not in datasets:
                raise AssertionError(
                    f"Unexpected dataset requested: {filename}"
                )

            dataframe = datasets[filename].copy()

            # Respect the production function's usecols argument.
            usecols = kwargs.get("usecols")
            if usecols is not None:
                dataframe = dataframe[usecols]

            return dataframe

        with patch(
            "src.intelligence.product_performance.pd.read_csv",
            side_effect=fake_read_csv,
        ):
            return build_product_performance()

    def test_only_delivered_items_are_counted(self):
        result = self.run_with_data()

        self.assertEqual(result["delivered_item_rows"], 3)
        self.assertEqual(result["delivered_order_count"], 2)
        self.assertAlmostEqual(
            result["delivered_item_sales"],
            60.0,
            places=2,
        )

    def test_product_aggregations_are_correct(self):
        result = self.run_with_data()
        products = result["product_performance"].set_index("product_id")

        self.assertAlmostEqual(products.loc["p1", "item_sales"], 40.0)
        self.assertEqual(products.loc["p1", "item_rows"], 2)
        self.assertEqual(products.loc["p1", "order_count"], 2)
        self.assertAlmostEqual(
            products.loc["p1", "average_item_price"],
            20.0,
        )

        self.assertAlmostEqual(products.loc["p2", "item_sales"], 20.0)
        self.assertEqual(products.loc["p2", "item_rows"], 1)
        self.assertEqual(products.loc["p2", "order_count"], 1)

    def test_category_sales_and_contributions_are_correct(self):
        result = self.run_with_data()
        categories = result["category_performance"].set_index("category")

        self.assertAlmostEqual(
            categories.loc["category_a", "item_sales"],
            40.0,
        )
        self.assertAlmostEqual(
            categories.loc["category_b", "item_sales"],
            20.0,
        )

        self.assertAlmostEqual(
            categories["sales_contribution_pct"].sum(),
            100.0,
            places=6,
        )

        self.assertEqual(
            categories.loc["category_a", "distinct_products"],
            1,
        )
        self.assertEqual(
            categories.loc["category_a", "order_count"],
            2,
        )

    def test_product_and_category_totals_match(self):
        result = self.run_with_data()

        product_total = result["product_performance"]["item_sales"].sum()
        category_total = result["category_performance"]["item_sales"].sum()

        self.assertAlmostEqual(
            product_total,
            result["delivered_item_sales"],
            places=2,
        )
        self.assertAlmostEqual(
            category_total,
            result["delivered_item_sales"],
            places=2,
        )

    def test_zero_sales_produce_zero_category_contributions(self):
        self.items["price"] = 0.0

        result = self.run_with_data()

        self.assertEqual(result["delivered_item_sales"], 0.0)
        self.assertTrue(
            (
                result["category_performance"]["sales_contribution_pct"]
                == 0.0
            ).all()
        )

    def test_negative_item_price_is_rejected(self):
        self.items.loc[0, "price"] = -1.0

        with self.assertRaisesRegex(ValueError, "Negative item prices"):
            self.run_with_data()

    def test_duplicate_order_ids_are_rejected(self):
        self.orders = pd.concat(
            [self.orders, self.orders.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaisesRegex(ValueError, "Duplicate order IDs"):
            self.run_with_data()


if __name__ == "__main__":
    unittest.main()
