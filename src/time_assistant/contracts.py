"""Typed domain schemas, DTOs, and service interfaces for time scheduling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol


def _require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _require_finite(
    value: float,
    field_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric.")
    num = float(value)
    if not math.isfinite(num):
        raise ValueError(f"{field_name} must be finite.")
    if minimum is not None and num < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}.")
    if maximum is not None and num > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}.")
    return num


@dataclass(frozen=True)
class TimeSlotRequest:
    """Validated request DTO for recommending an activity in a daily block."""

    user_id: str
    hour_of_day: int
    energy_level: float
    focus_score: float
    available_minutes: int = 60
    excluded_activities: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_id", _require_non_empty(self.user_id, "user_id"))
        if isinstance(self.hour_of_day, bool) or not isinstance(self.hour_of_day, int):
            raise TypeError("hour_of_day must be an integer.")
        if self.hour_of_day < 0 or self.hour_of_day > 23:
            raise ValueError("hour_of_day must be between 0 and 23.")
        _require_finite(self.energy_level, "energy_level", minimum=0.0, maximum=1.0)
        _require_finite(self.focus_score, "focus_score", minimum=0.0, maximum=1.0)
        if self.available_minutes < 5 or self.available_minutes > 1440:
            raise ValueError("available_minutes must be between 5 and 1440.")


@dataclass(frozen=True)
class ActivityPolicyConfig:
    """Configuration governing reward weights across ATUS activity classes."""

    productivity_weight: float = 0.55
    wellbeing_weight: float = 0.30
    balance_weight: float = 0.15

    def __post_init__(self) -> None:
        p = _require_finite(self.productivity_weight, "productivity_weight", minimum=0.0, maximum=1.0)
        w = _require_finite(self.wellbeing_weight, "wellbeing_weight", minimum=0.0, maximum=1.0)
        b = _require_finite(self.balance_weight, "balance_weight", minimum=0.0, maximum=1.0)
        if p + w + b <= 0.0:
            raise ValueError("At least one policy weight must be > 0.")


@dataclass(frozen=True)
class ScheduleAllocation:
    """Single recommended time block allocation."""

    activity_id: int
    activity_name: str
    duration_minutes: int
    expected_reward: float


@dataclass(frozen=True)
class RecommendationPlan:
    """Aggregated recommendation plan with compute telemetry."""

    user_id: str
    hour_of_day: int
    allocations: tuple[ScheduleAllocation, ...]
    total_reward: float
    latency_ms: float
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LatencySummary:
    """Percentile compute latency report from synthetic load verification."""

    iterations: int
    min_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_ops_sec: float


class PolicyScorer(Protocol):
    """Interface for scoring ATUS activities for a user state."""

    def score_activities(
        self,
        request: TimeSlotRequest,
        catalog: dict[int, str],
        policy: ActivityPolicyConfig,
    ) -> list[tuple[int, str, float]]:
        ...


class ScheduleOptimizer(Protocol):
    """Interface for packing available minutes using ranked activity scores."""

    def build_plan(
        self,
        request: TimeSlotRequest,
        scored: list[tuple[int, str, float]],
    ) -> list[ScheduleAllocation]:
        ...


class TelemetrySink(Protocol):
    """Decoupled observability sink for recording decision latency and counts."""

    def record_decision(self, user_id: str, latency_ms: float, reward: float) -> None:
        ...
