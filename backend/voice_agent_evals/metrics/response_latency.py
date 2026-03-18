"""
Response Latency Metric (Tier 2 — Algorithmic)

Measures the time between the user finishing speaking and the agent starting to speak.
Operates on timing data from VoiceTrace turns; returns `no_data` when timing is absent.

Thresholds:
  Mean < 800ms    → excellent (1.0)
  Mean 800–1500ms → good (0.75)
  Mean 1500–2500ms → fair (0.5)
  Mean > 2500ms   → poor (0.25)
"""

from __future__ import annotations

import statistics
from typing import Any

from voice_agent_evals.metrics.base import BaseMetric, MetricResult, MetricScore
from voice_agent_evals.trace import Speaker, VoiceTrace


class ResponseLatencyMetric(BaseMetric):
    name = "response_latency"

    def evaluate(self, trace: VoiceTrace) -> MetricResult:
        latencies = _compute_latencies(trace)

        if not latencies:
            return MetricResult(
                metric_name=self.name,
                trace_id=trace.trace_id,
                score=MetricScore(
                    score=0.0,
                    label="no_data",
                    reasoning="No adjacent (user → agent) turn pairs with complete timing found.",
                    details={"latencies_ms": []},
                ),
            )

        mean_ms = statistics.mean(latencies)
        p50_ms = statistics.median(latencies)
        p95_ms = _percentile(latencies, 95)

        score, label = _score_from_mean(mean_ms)

        reasoning = (
            f"Mean response latency is {mean_ms:.0f}ms "
            f"(P50={p50_ms:.0f}ms, P95={p95_ms:.0f}ms) across {len(latencies)} turn pair(s)."
        )

        details: dict[str, Any] = {
            "latencies_ms": [round(v, 1) for v in latencies],
            "mean_ms": round(mean_ms, 1),
            "p50_ms": round(p50_ms, 1),
            "p95_ms": round(p95_ms, 1),
            "n_pairs": len(latencies),
        }

        return MetricResult(
            metric_name=self.name,
            trace_id=trace.trace_id,
            score=MetricScore(score=score, label=label, reasoning=reasoning, details=details),
        )


def _compute_latencies(trace: VoiceTrace) -> list[float]:
    """Return list of response latencies (ms) for each user→agent adjacent pair."""
    latencies: list[float] = []
    turns = trace.turns
    for i in range(len(turns) - 1):
        curr = turns[i]
        nxt = turns[i + 1]
        if curr.speaker != Speaker.USER or nxt.speaker != Speaker.AGENT:
            continue
        if (
            curr.timing is None
            or curr.timing.speech_end_ms is None
            or nxt.timing is None
            or nxt.timing.speech_start_ms is None
        ):
            continue
        latency = nxt.timing.speech_start_ms - curr.timing.speech_end_ms
        latencies.append(latency)
    return latencies


def _percentile(data: list[float], pct: int) -> float:
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def _score_from_mean(mean_ms: float) -> tuple[float, str]:
    if mean_ms < 800:
        return 1.0, "excellent"
    if mean_ms < 1500:
        return 0.75, "good"
    if mean_ms < 2500:
        return 0.5, "fair"
    return 0.25, "poor"
