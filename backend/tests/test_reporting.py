from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from reporting import (
    build_business_insight_payload,
    build_report_snapshot_svg,
    export_records_csv_bytes,
    read_recent_log_events,
)


class ReportingTests(unittest.TestCase):
    def test_business_payload_aggregates_top_pain_points(self) -> None:
        payload = build_business_insight_payload(
            [
                {"sentiment": "negative", "confidence": 0.9, "pain_points": ["包装破损", "物流慢"]},
                {"sentiment": "negative", "confidence": 0.8, "pain_points": ["包装破损"]},
                {"sentiment": "positive", "confidence": 0.95, "pain_points": []},
            ],
            estimated_orders_per_month=1000,
            average_order_value=100,
            return_loss_rate=0.2,
            language="zh",
        )

        self.assertEqual(payload["metrics"]["analyzed_count"], 3)
        self.assertEqual(payload["metrics"]["sentiment_counts"]["negative"], 2)
        self.assertEqual(payload["top_pain_points"][0]["pain_point"], "包装破损")
        self.assertEqual(payload["top_pain_points"][0]["count"], 2)
        self.assertGreater(payload["top_pain_points"][0]["estimated_monthly_loss"], 0)

    def test_export_csv_and_svg_snapshot_are_downloadable(self) -> None:
        records = [{"index": 1, "sentiment": "negative", "pain_points": ["物流慢"], "summary_zh": "物流慢"}]
        csv_bytes = export_records_csv_bytes(records)
        self.assertTrue(csv_bytes.startswith(b"\xef\xbb\xbf"))
        self.assertIn("sentiment", csv_bytes.decode("utf-8-sig"))

        payload = build_business_insight_payload(records, language="zh")
        svg = build_report_snapshot_svg(payload)
        self.assertIn("<svg", svg)
        self.assertIn("物流慢", svg)

    def test_read_recent_log_events_filters_status(self) -> None:
        log_path = Path.cwd() / "tmp" / "unit_reporting_logs.jsonl"
        log_path.parent.mkdir(exist_ok=True)
        log_path.write_text(
            "\n".join(
                [
                    json.dumps({"status": "ok", "request_id": "a"}),
                    json.dumps({"status": "error", "request_id": "b"}),
                ]
            ),
            encoding="utf-8",
        )

        events = read_recent_log_events(limit=10, status="error", log_path=log_path)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["request_id"], "b")


if __name__ == "__main__":
    unittest.main()
