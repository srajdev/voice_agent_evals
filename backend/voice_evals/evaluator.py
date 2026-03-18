"""
Evaluator orchestrator — runs all metrics against a VoiceTrace, produces a JSON report.

Usage:
    from voice_evals.evaluator import Evaluator
    from voice_evals.trace import VoiceTrace

    evaluator = Evaluator()
    report = evaluator.run(trace)
    print(report.model_dump_json(indent=2))
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from voice_evals.metrics.base import BaseMetric, MetricResult
from voice_evals.metrics.coherence import CoherenceMetric
from voice_evals.metrics.conversation_quality import ConversationQualityMetric
from voice_evals.metrics.intent import IntentAccuracyMetric
from voice_evals.metrics.response_latency import ResponseLatencyMetric
from voice_evals.metrics.speech_style import (
    EmpathyMetric,
    VerbosityMatchMetric,
    VocabularyMatchMetric,
)
from voice_evals.metrics.task_completion import TaskCompletionMetric
from voice_evals.metrics.vad_quality import InterruptionRecoveryMetric, VadQualityMetric
from voice_evals.trace import VoiceTrace


class EvaluationReport(BaseModel):
    """Full evaluation report for one VoiceTrace."""

    report_id: str
    trace_id: str
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: float

    # Per-metric results
    results: list[MetricResult]

    # Rolled-up summary
    summary: dict[str, Any] = Field(default_factory=dict)

    # Source trace (embedded for portability)
    trace: VoiceTrace | None = None

    def get_score(self, metric_name: str) -> float | None:
        for r in self.results:
            if r.metric_name == metric_name:
                return r.score.score
        return None

    @property
    def overall_score(self) -> float:
        """Simple average of all metric scores (excluding errors and no_data)."""
        valid = [
            r.score.score
            for r in self.results
            if r.score.label not in {"error", "no_data", "not_applicable"}
        ]
        return sum(valid) / len(valid) if valid else 0.0


TIER_METRICS: dict[int, list[type[BaseMetric]]] = {
    1: [ConversationQualityMetric, CoherenceMetric, IntentAccuracyMetric, TaskCompletionMetric],
    2: [ResponseLatencyMetric, VadQualityMetric, InterruptionRecoveryMetric],
    3: [VerbosityMatchMetric, EmpathyMetric, VocabularyMatchMetric],
}

DEFAULT_METRICS: list[type[BaseMetric]] = [
    metric for tier_metrics in TIER_METRICS.values() for metric in tier_metrics
]


class Evaluator:
    """
    Orchestrates running metrics against a VoiceTrace.

    Args:
        metrics: List of metric classes to run. Defaults to all Tier 1 metrics.
        embed_trace: Whether to embed the full trace in the report (useful for standalone reports).
    """

    def __init__(
        self,
        metrics: list[type[BaseMetric]] | None = None,
        embed_trace: bool = False,
    ):
        metric_classes = metrics or DEFAULT_METRICS
        self.metrics: list[BaseMetric] = [cls() for cls in metric_classes]
        self.embed_trace = embed_trace

    def run(self, trace: VoiceTrace) -> EvaluationReport:
        """Run all metrics and return a complete EvaluationReport."""
        import uuid

        start = time.perf_counter()
        results: list[MetricResult] = []

        for metric in self.metrics:
            try:
                result = metric.evaluate(trace)
                results.append(result)
            except Exception as e:
                # Don't let one metric failure kill the entire evaluation
                from voice_evals.metrics.base import MetricResult, MetricScore
                results.append(
                    MetricResult(
                        metric_name=metric.name,
                        trace_id=trace.trace_id,
                        score=MetricScore(
                            score=0.0,
                            label="error",
                            reasoning=f"Metric raised exception: {e}",
                        ),
                    )
                )

        elapsed_ms = (time.perf_counter() - start) * 1000

        report = EvaluationReport(
            report_id=str(uuid.uuid4()),
            trace_id=trace.trace_id,
            duration_ms=elapsed_ms,
            results=results,
            trace=trace if self.embed_trace else None,
        )

        report.summary = _build_summary(trace, report)
        return report


def _build_summary(trace: VoiceTrace, report: EvaluationReport) -> dict[str, Any]:
    return {
        "overall_score": round(report.overall_score, 3),
        "overall_label": _score_to_label(report.overall_score),
        "n_turns": len(trace.turns),
        "n_user_turns": len(trace.user_turns),
        "n_agent_turns": len(trace.agent_turns),
        "duration_ms": trace.duration_ms,
        "platform": trace.platform_info.platform,
        "scenario": trace.scenario.scenario_id if trace.scenario else None,
        "metrics": {
            r.metric_name: {
                "score": round(r.score.score, 3),
                "label": r.score.label,
            }
            for r in report.results
        },
    }


def _score_to_label(score: float) -> str:
    if score >= 0.85:
        return "excellent"
    if score >= 0.70:
        return "good"
    if score >= 0.50:
        return "fair"
    return "poor"
