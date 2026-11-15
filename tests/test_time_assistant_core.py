"""Comprehensive domain, validation, DI, and load tests for time_assistant."""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.time_assistant.contracts import (
    ActivityPolicyConfig,
    PolicyScorer,
    TimeSlotRequest,
)
from src.time_assistant.scheduler import (
    InMemoryTelemetrySink,
    ServiceContainer,
    TimeAssistantEngine,
    load_activity_catalog,
    run_synthetic_schedule_benchmark,
)


class _StubScorer:
    def score_activities(
        self,
        request: TimeSlotRequest,
        catalog: dict[int, str],
        policy: ActivityPolicyConfig,
    ) -> list[tuple[int, str, float]]:
        return [(11, catalog[11], 0.99)]


class TestTimeAssistantArchitecture(unittest.TestCase):
    def test_catalog_contains_twelve_atus_categories(self) -> None:
        catalog = load_activity_catalog()
        self.assertEqual(len(catalog), 12)
        self.assertEqual(catalog[7], "Sleeping")
        self.assertEqual(catalog[11], "Working")

    def test_dto_validation_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            TimeSlotRequest(user_id=" ", hour_of_day=10, energy_level=0.5, focus_score=0.5)
        with self.assertRaises(ValueError):
            TimeSlotRequest(user_id="u1", hour_of_day=24, energy_level=0.5, focus_score=0.5)
        with self.assertRaises(ValueError):
            TimeSlotRequest(user_id="u1", hour_of_day=10, energy_level=math.nan, focus_score=0.5)
        with self.assertRaises(ValueError):
            TimeSlotRequest(
                user_id="u1",
                hour_of_day=10,
                energy_level=0.5,
                focus_score=0.5,
                available_minutes=2,
            )
        with self.assertRaises(ValueError):
            ActivityPolicyConfig(productivity_weight=0.0, wellbeing_weight=0.0, balance_weight=0.0)

    def test_morning_vs_night_recommendation_semantics(self) -> None:
        telemetry = InMemoryTelemetrySink()
        engine = TimeAssistantEngine.create_default(telemetry=telemetry)

        morning_plan = engine.recommend(
            TimeSlotRequest(user_id="u-morning", hour_of_day=10, energy_level=0.9, focus_score=0.95)
        )
        night_plan = engine.recommend(
            TimeSlotRequest(user_id="u-night", hour_of_day=23, energy_level=0.2, focus_score=0.2)
        )
        self.assertIn(morning_plan.allocations[0].activity_id, {2, 11})
        self.assertEqual(night_plan.allocations[0].activity_id, 7)
        self.assertEqual(len(telemetry.events), 2)

    def test_dependency_injection_and_exclusions(self) -> None:
        container = ServiceContainer()
        container.register_instance(PolicyScorer, _StubScorer())
        scorer = container.resolve(PolicyScorer)
        telemetry = InMemoryTelemetrySink()
        engine = TimeAssistantEngine(
            scorer=scorer,
            optimizer=TimeAssistantEngine.create_default()._optimizer,
            telemetry=telemetry,
        )
        plan = engine.recommend(
            TimeSlotRequest(user_id="u-di", hour_of_day=14, energy_level=0.6, focus_score=0.6, available_minutes=90)
        )
        self.assertEqual(plan.allocations[0].activity_id, 11)
        self.assertEqual(plan.allocations[0].duration_minutes, 60)

        all_ids = tuple(range(12))
        with self.assertRaises(ValueError):
            TimeAssistantEngine.create_default().recommend(
                TimeSlotRequest(
                    user_id="u-empty",
                    hour_of_day=12,
                    energy_level=0.5,
                    focus_score=0.5,
                    excluded_activities=all_ids,
                )
            )

    def test_synthetic_schedule_load_benchmark(self) -> None:
        engine = TimeAssistantEngine.create_default()
        summary = run_synthetic_schedule_benchmark(engine=engine, iterations=300, max_p95_ms=20.0)
        self.assertEqual(summary.iterations, 300)
        self.assertLess(summary.p95_ms, 20.0)
        self.assertGreater(summary.throughput_ops_sec, 500.0)


if __name__ == "__main__":
    unittest.main()
