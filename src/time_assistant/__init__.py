"""Clean-architecture domain core for RL time-use scheduling and telemetry."""

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
from src.time_assistant.scheduler import (
    CircadianPolicyScorer,
    GreedyScheduleOptimizer,
    InMemoryTelemetrySink,
    ServiceContainer,
    TimeAssistantEngine,
    load_activity_catalog,
    run_synthetic_schedule_benchmark,
)

__all__ = [
    "ActivityPolicyConfig",
    "CircadianPolicyScorer",
    "GreedyScheduleOptimizer",
    "InMemoryTelemetrySink",
    "LatencySummary",
    "PolicyScorer",
    "RecommendationPlan",
    "ScheduleAllocation",
    "ScheduleOptimizer",
    "ServiceContainer",
    "TelemetrySink",
    "TimeAssistantEngine",
    "TimeSlotRequest",
    "load_activity_catalog",
    "run_synthetic_schedule_benchmark",
]
