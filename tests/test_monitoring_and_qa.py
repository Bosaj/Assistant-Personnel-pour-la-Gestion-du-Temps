"""Monitoring, health probe, and QA harness tests."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from monitoring.health import check_liveness, check_readiness, get_health_status
from monitoring.metrics import record_request, set_health_status, update_eval_metric
from scripts.eval_harness import run_evaluation


class TestMonitoringAndQA(unittest.TestCase):
    def test_liveness_and_readiness(self) -> None:
        self.assertIs(check_liveness(), True)
        self.assertIs(check_readiness(), True)

    def test_health_status(self) -> None:
        status = get_health_status()
        self.assertEqual(status["status"], "UP")
        self.assertEqual(status["service"], "Assistant-Personnel-pour-la-Gestion-du-Temps")
        self.assertTrue(status["checks"]["liveness"])
        self.assertTrue(status["checks"]["readiness"])

    def test_eval_harness(self) -> None:
        results = run_evaluation()
        self.assertEqual(results["status"], "PASSED")
        self.assertIn("quality_index", results["metrics"])
        self.assertIn("p95_latency_ms", results["metrics"])
        record_request("GET", "/health", "200", 0.001)
        update_eval_metric("test", 0.99)
        set_health_status(True)


if __name__ == "__main__":
    unittest.main()
