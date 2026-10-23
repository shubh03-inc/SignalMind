
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
SRC = PROJECT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intelligence.revenue_intelligence import build_revenue_intelligence


REPORTS = PROJECT / "reports"
REPORTS.mkdir(exist_ok=True)


def main():
    results = build_revenue_intelligence()

    reconciliation = results["reconciliation"]
    status_summary = results["status_summary"]
    mismatches = results["mismatches"]
    missing_items = results["missing_items_with_payments"]
    delivered_differences = results["delivered_differences"]

    # Export detailed and summarized results.
    reconciliation.to_csv(
        REPORTS / "revenue_reconciliation.csv",
        index=False,
    )

    status_summary.to_csv(
        REPORTS / "revenue_by_order_status.csv",
        index=False,
    )

    mismatches.to_csv(
        REPORTS / "revenue_mismatches.csv",
        index=False,
    )

    missing_items.to_csv(
        REPORTS / "revenue_missing_items_with_payments.csv",
        index=False,
    )

    delivered_differences.to_csv(
        REPORTS / "revenue_delivered_differences.csv",
        index=False,
    )

    # Print the summary.
    print("=" * 65)
    print("SIGNALMIND REVENUE REPORT")
    print("=" * 65)

    print("Orders checked:", results["total_orders"])
    print("Orders with item records:", results["orders_with_items"])
    print("Orders without item records:", results["orders_without_items"])
    print("Orders with payment records:", results["orders_with_payments"])
    print("Orders without payment records:", results["orders_without_payments"])
    print(
        "Orders with payment differences over R$1:",
        results["orders_with_large_differences"],
    )
    print(
        "Delivered orders with differences over R$1:",
        results["delivered_orders_with_large_differences"],
    )

    print("\nTOTALS BY ORDER STATUS")
    print(status_summary.round(2).to_string(index=False))

    print("\nLARGEST PAYMENT DIFFERENCES")
    display_columns = [
        "order_id",
        "order_status",
        "item_records",
        "payment_records",
        "item_plus_freight",
        "payment_total",
        "difference",
        "issue",
    ]

    if mismatches.empty:
        print("No flagged orders.")
    else:
        print(
            mismatches.assign(
                absolute_difference=mismatches["difference"].abs()
            )
            .sort_values("absolute_difference", ascending=False)
            .head(10)[display_columns]
            .round(2)
            .to_string(index=False)
        )

    print("\nSaved reports:")
    print("- reports/revenue_reconciliation.csv")
    print("- reports/revenue_by_order_status.csv")
    print("- reports/revenue_mismatches.csv")
    print("- reports/revenue_missing_items_with_payments.csv")
    print("- reports/revenue_delivered_differences.csv")

    print(
        "\nNote: flagged differences are investigation signals, "
        "not confirmed revenue losses."
    )


if __name__ == "__main__":
    main()
