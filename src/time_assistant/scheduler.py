"""DI container, circadian activity scorer, schedule optimizer, and load profiler."""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.time_assistant.contracts import (
    ActivityPolicyConfig,
    LatencySummary,
    PolicyScorer,
    RecommendationPlan,
    ScheduleAllocation,
    ScheduleOptimizer,
    TelemetrySink,
    TimeSlotRequest,
)

_DEFAULT_CATALOG: dict[int, str] = {
    0: "Caring for Household Members",
    1: "Eating and Drinking",
    2: "Education",
    3: "Household Activities",
    4: "Personal Care",
    5: "Religious and Volunteer Activities",
    6: "Shopping",
    7: "Sleeping",
    8: "Socializing and Leisure",
    9: "Sports and Exercise",
    10: "Traveling / Commuting",
    11: "Working",
}

# (productivity_base, wellbeing_base, preferred_hour_center, hour_spread)
_ACTIVITY_PRIORS: dict[int, tuple[float, float, float, float]] = {
    0: (0.45, 0.70, 18.0, 4.0),
    1: (0.30, 0.85, 13.0, 4.5),
    2: (0.92, 0.55, 10.5, 4.0),
    3: (0.55, 0.45, 17.0, 4.0),
    4: (0.35, 0.85, 7.5, 3.0),
    5: (0.40, 0.80, 11.0, 5.0),
    6: (0.45, 0.50, 16.0, 4.0),
    7: (0.10, 0.98, 2.0, 3.5),
    8: (0.25, 0.90, 20.0, 3.5),
    9: (0.60, 0.95, 7.5, 4.5),
    10: (0.40, 0.25, 8.5, 3.0),
    11: (0.96, 0.45, 11.0, 4.5),
}


def load_activity_catalog(catalog_path: str | Path | None = None) -> dict[int, str]:
    """Load ATUS activity labels from model/activity_names.json with safe fallback."""
    target = (
        Path(catalog_path)
        if catalog_path is not None
        else Path(__file__).resolve().parents[2] / "model" / "activity_names.json"
    )
    if not target.exists():
        return dict(_DEFAULT_CATALOG)
    with open(target, encoding="utf-8") as handle:
        raw = json.load(handle)
    return {int(k): str(v) for k, v in raw.items()}


class InMemoryTelemetrySink:
    """Thread-safe decision telemetry accumulator for tests and monitoring."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.events: list[tuple[str, float, float]] = []

    def record_decision(self, user_id: str, latency_ms: float, reward: float) -> None:
        with self._lock:
            self.events.append((user_id, float(latency_ms), float(reward)))


class CircadianPolicyScorer:
    """Fast pure-Python scorer combining circadian affinity, user energy, and focus."""

    def score_activities(
        self,
        request: TimeSlotRequest,
        catalog: dict[int, str],
        policy: ActivityPolicyConfig,
    ) -> list[tuple[int, str, float]]:
        excluded = set(request.excluded_activities)
        scored: list[tuple[int, str, float]] = []
        for aid, label in catalog.items():
            if aid in excluded:
                continue
            prod, well, center, spread = _ACTIVITY_PRIORS.get(aid, (0.5, 0.5, 12.0, 5.0))
            hour_diff = min(abs(request.hour_of_day - center), 24.0 - abs(request.hour_of_day - center))
            circadian_fit = math.exp(-((hour_diff / spread) ** 2))
            if request.hour_of_day >= 22 or request.hour_of_day <= 5:
                sleep_bonus = 0.6 if aid == 7 else 0.0
            else:
                sleep_bonus = 0.0
            energy_fit = prod * request.focus_score * 0.55 + prod * request.energy_level * 0.45
            utility = (
                policy.productivity_weight * energy_fit
                + policy.wellbeing_weight * well
                + policy.balance_weight * circadian_fit
                + sleep_bonus
            )
            scored.append((aid, label, round(utility, 6)))
        scored.sort(key=lambda item: (-item[2], item[0]))
        return scored


class GreedyScheduleOptimizer:
    """Packs available time window into bounded 15-60 minute high-utility blocks."""

    def __init__(self, max_block_minutes: int = 60, min_block_minutes: int = 15) -> None:
        self._max_block = max_block_minutes
        self._min_block = min_block_minutes

    def build_plan(
        self,
        request: TimeSlotRequest,
        scored: list[tuple[int, str, float]],
    ) -> list[ScheduleAllocation]:
        if not scored:
            raise ValueError("No eligible activities remain to build a schedule plan.")
        remaining = request.available_minutes
        allocations: list[ScheduleAllocation] = []
        for aid, name, utility in scored:
            if remaining <= 0:
                break
            block = min(remaining, self._max_block)
            if block < self._min_block and allocations:
                break
            allocations.append(
                ScheduleAllocation(
                    activity_id=aid,
                    activity_name=name,
                    duration_minutes=block,
                    expected_reward=round(utility * (block / 60.0), 6),
                )
            )
            remaining -= block
        return allocations


class ServiceContainer:
    """Minimal dependency injection container for scheduling services."""

    def __init__(self) -> None:
        self._instances: dict[type[Any], Any] = {}
        self._factories: dict[type[Any], Callable[[], Any]] = {}

    def register_instance(self, interface: type[Any], instance: Any) -> None:
        self._instances[interface] = instance

    def register_factory(self, interface: type[Any], factory: Callable[[], Any]) -> None:
        self._factories[interface] = factory

    def resolve(self, interface: type[Any]) -> Any:
        if interface in self._instances:
            return self._instances[interface]
        if interface not in self._factories:
            raise KeyError(f"No provider registered for {interface.__name__}.")
        obj = self._factories[interface]()
        self._instances[interface] = obj
        return obj


class TimeAssistantEngine:
    """Orchestrates activity scoring, schedule optimization, and telemetry."""

    def __init__(
        self,
        scorer: PolicyScorer,
        optimizer: ScheduleOptimizer,
        telemetry: TelemetrySink,
        catalog: dict[int, str] | None = None,
    ) -> None:
        self._scorer = scorer
        self._optimizer = optimizer
        self._telemetry = telemetry
        self._catalog = catalog if catalog is not None else load_activity_catalog()

    @classmethod
    def create_default(cls, telemetry: TelemetrySink | None = None) -> TimeAssistantEngine:
        container = ServiceContainer()
        container.register_instance(PolicyScorer, CircadianPolicyScorer())
        container.register_instance(ScheduleOptimizer, GreedyScheduleOptimizer())
        container.register_instance(TelemetrySink, telemetry or InMemoryTelemetrySink())
        return cls(
            scorer=container.resolve(PolicyScorer),
            optimizer=container.resolve(ScheduleOptimizer),
            telemetry=container.resolve(TelemetrySink),
        )

    def recommend(
        self,
        request: TimeSlotRequest,
        policy: ActivityPolicyConfig | None = None,
    ) -> RecommendationPlan:
        t0 = time.perf_counter()
        cfg = policy or ActivityPolicyConfig()
        scored = self._scorer.score_activities(request=request, catalog=self._catalog, policy=cfg)
        allocations = self._optimizer.build_plan(request=request, scored=scored)
        total_reward = round(sum(item.expected_reward for item in allocations), 6)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._telemetry.record_decision(request.user_id, latency_ms, total_reward)
        return RecommendationPlan(
            user_id=request.user_id,
            hour_of_day=request.hour_of_day,
            allocations=tuple(allocations),
            total_reward=total_reward,
            latency_ms=latency_ms,
            metadata={"catalog_size": str(len(self._catalog))},
        )


def _percentile(ordered: list[float], pct: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def run_synthetic_schedule_benchmark(
    engine: TimeAssistantEngine,
    iterations: int = 250,
    max_p95_ms: float = 20.0,
) -> LatencySummary:
    """Run synthetic circadian scheduling load and enforce p95 latency SLA."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1.")
    samples: list[float] = []
    suite_start = time.perf_counter()
    for idx in range(iterations):
        req = TimeSlotRequest(
            user_id=f"user-{idx % 12}",
            hour_of_day=idx % 24,
            energy_level=0.75,
            focus_score=0.80,
            available_minutes=120,
        )
        t0 = time.perf_counter()
        engine.recommend(req)
        samples.append((time.perf_counter() - t0) * 1000.0)
    total_ms = max((time.perf_counter() - suite_start) * 1000.0, 1e-6)
    ordered = sorted(samples)
    p95 = _percentile(ordered, 95.0)
    if p95 > max_p95_ms:
        raise TimeoutError(f"Schedule engine p95 {p95:.3f}ms exceeded {max_p95_ms:.3f}ms.")
    return LatencySummary(
        iterations=iterations,
        min_ms=ordered[0],
        p50_ms=_percentile(ordered, 50.0),
        p95_ms=p95,
        p99_ms=_percentile(ordered, 99.0),
        throughput_ops_sec=(iterations * 1000.0) / total_ms,
    )
