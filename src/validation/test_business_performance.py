import math
import unittest
from numbers import Real

from src.intelligence.sales_intelligence import build_sales_intelligence
from src.intelligence.revenue_intelligence import build_revenue_intelligence
from src.intelligence.product_performance import build_product_performance
from src.intelligence.business_performance import build_business_performance


class TestBusinessPerformance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sales = build_sales_intelligence()
        cls.revenue = build_revenue_intelligence()
        cls.products = build_product_performance()
        cls.business = build_business_performance()

    def test_delivered_order_counts_match(self):
        self.assertEqual(
            self.sales["total_delivered_orders"],
            self.products["delivered_order_count"],
        )
        self.assertEqual(
            self.sales["total_delivered_orders"],
            self.business["total_delivered_orders"],
        )

    def test_delivered_order_share_is_consistent(self):
        total_orders = self.business["total_orders"]
        delivered_orders = self.business["total_delivered_orders"]

        expected_share = (
            delivered_orders / total_orders * 100
            if total_orders > 0
            else 0.0
        )

        self.assertAlmostEqual(
            self.business["delivered_order_share_percent"],
            expected_share,
            places=6,
        )

    def test_delivered_sales_match_business_performance(self):
        self.assertAlmostEqual(
            self.sales["item_sales"],
            self.business["item_sales"],
            places=2,
        )

    def test_delivered_sales_match_product_performance(self):
        self.assertAlmostEqual(
            self.sales["item_sales"],
            self.products["delivered_item_sales"],
            places=2,
        )

    def test_freight_matches_business_performance(self):
        self.assertAlmostEqual(
            self.sales["freight_total"],
            self.business["freight_total"],
            places=2,
        )

    def test_reconciliation_counts_match(self):
        self.assertEqual(
            self.revenue["total_orders"],
            self.business["total_orders"],
        )
        self.assertEqual(
            self.revenue["orders_without_items"],
            self.business["orders_without_items"],
        )
        self.assertEqual(
            self.revenue["orders_without_payments"],
            self.business["orders_without_payments"],
        )
        self.assertEqual(
            self.revenue["orders_with_large_differences"],
            self.business["orders_with_large_differences"],
        )

    def test_reconciliation_summary_covers_all_orders(self):
        summary_orders = self.revenue["reconciliation_summary"][
            "order_count"
        ].sum()

        self.assertEqual(
            summary_orders,
            self.revenue["total_orders"],
        )

    def test_required_business_result_keys_exist(self):
        required_keys = {
            "total_orders",
            "total_delivered_orders",
            "delivered_order_share_percent",
            "item_sales",
            "freight_total",
            "monthly_sales",
            "category_sales",
            "orders_without_items",
            "orders_without_payments",
            "orders_with_large_differences",
            "large_difference_rate_percent",
            "delivered_orders_with_large_differences",
            "reconciliation_summary",
            "status_summary",
            "top_product",
            "top_category",
            "unmatched_order_items",
            "unmatched_product_rows",
            "insights",
        }

        self.assertTrue(
            required_keys.issubset(self.business.keys()),
            f"Missing keys: {required_keys - self.business.keys()}",
        )

    def test_scalar_metrics_are_finite(self):
        results = {
            "sales": self.sales,
            "revenue": self.revenue,
            "products": self.products,
            "business": self.business,
        }

        for report_name, result in results.items():
            for key, value in result.items():
                if isinstance(value, Real) and not isinstance(value, bool):
                    with self.subTest(report=report_name, metric=key):
                        self.assertTrue(
                            math.isfinite(value),
                            f"{report_name}.{key} is not finite: {value}",
                        )

    def test_numeric_dataframes_contain_no_infinity(self):
        dataframes = {
            "sales.monthly_sales": self.sales["monthly_sales"],
            "sales.category_sales": self.sales["category_sales"],
            "revenue.reconciliation": self.revenue["reconciliation"],
            "revenue.status_summary": self.revenue["status_summary"],
            "revenue.reconciliation_summary": (
                self.revenue["reconciliation_summary"]
            ),
            "revenue.mismatches": self.revenue["mismatches"],
            "revenue.missing_items_with_payments": (
                self.revenue["missing_items_with_payments"]
            ),
            "revenue.delivered_differences": (
                self.revenue["delivered_differences"]
            ),
            "products.product_performance": (
                self.products["product_performance"]
            ),
            "products.category_performance": (
                self.products["category_performance"]
            ),
            "business.monthly_sales": self.business["monthly_sales"],
            "business.category_sales": self.business["category_sales"],
            "business.reconciliation_summary": (
                self.business["reconciliation_summary"]
            ),
            "business.status_summary": self.business["status_summary"],
        }

        for name, dataframe in dataframes.items():
            numeric_data = dataframe.select_dtypes(include="number")

            with self.subTest(dataframe=name):
                self.assertFalse(
                    numeric_data.isin(
                        [float("inf"), float("-inf")]
                    ).any().any(),
                    f"{name} contains infinite numeric values",
                )


if __name__ == "__main__":
    unittest.main()
