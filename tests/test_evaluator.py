"""Tests for the Evaluator — uses mocked LLM calls."""

from unittest.mock import MagicMock, patch

import pytest

from voice_evals.evaluator import Evaluator, _score_to_label
from voice_evals.metrics.base import MetricResult, MetricScore
from voice_evals.trace import ScenarioConfig, Speaker, Turn, VoiceTrace


def make_trace_with_turns() -> VoiceTrace:
    trace = VoiceTrace(
        scenario=ScenarioConfig(
            scenario_id="eval-test",
            expected_task="Order a pizza",
            completion_criteria="Agent confirms pizza order with toppings and delivery address",
            expected_intents=["request_order", "provide_toppings"],
        )
    )
    trace.add_turn(Turn(speaker=Speaker.USER, transcript="I'd like to order a pizza please."))
    trace.add_turn(Turn(speaker=Speaker.AGENT, transcript="What toppings would you like?"))
    trace.add_turn(Turn(speaker=Speaker.USER, transcript="Pepperoni and mushrooms."))
    trace.add_turn(Turn(speaker=Speaker.AGENT, transcript="Got it. What's your delivery address?"))
    return trace


MOCK_GOOD_RESPONSE = """
{
  "score": 0.85,
  "label": "good",
  "reasoning": "The agent communicates clearly and naturally.",
  "issues": [],
  "strengths": ["Short responses", "Natural phrasing"]
}
"""


@pytest.fixture
def patched_llm():
    with patch("voice_evals.metrics.base.call_llm_judge", return_value=MOCK_GOOD_RESPONSE):
        yield


def test_evaluator_runs_all_metrics(patched_llm):
    trace = make_trace_with_turns()
    evaluator = Evaluator()
    report = evaluator.run(trace)

    assert report.trace_id == trace.trace_id
    assert len(report.results) == 10  # 4 Tier 1 + 3 Tier 2 + 3 Tier 3 metrics
    metric_names = {r.metric_name for r in report.results}
    # Tier 1
    assert "conversation_quality" in metric_names
    assert "coherence" in metric_names
    assert "intent_accuracy" in metric_names
    assert "task_completion" in metric_names
    # Tier 2
    assert "response_latency" in metric_names
    assert "vad_quality" in metric_names
    assert "interruption_recovery" in metric_names
    # Tier 3
    assert "verbosity_match" in metric_names
    assert "empathy" in metric_names
    assert "vocabulary_match" in metric_names


def test_evaluator_overall_score(patched_llm):
    trace = make_trace_with_turns()
    evaluator = Evaluator()
    report = evaluator.run(trace)
    assert 0.0 <= report.overall_score <= 1.0


def test_evaluator_summary(patched_llm):
    trace = make_trace_with_turns()
    evaluator = Evaluator()
    report = evaluator.run(trace)
    s = report.summary
    assert s["n_turns"] == 4
    assert s["n_user_turns"] == 2
    assert s["n_agent_turns"] == 2
    assert "overall_score" in s
    assert "metrics" in s


def test_evaluator_metric_failure_is_caught():
    """A crashing metric should not kill the entire evaluation."""
    from voice_evals.metrics.base import BaseMetric

    class BrokenMetric(BaseMetric):
        name = "broken"

        def evaluate(self, trace):
            raise RuntimeError("Simulated metric failure")

    with patch("voice_evals.metrics.base.call_llm_judge", return_value=MOCK_GOOD_RESPONSE):
        trace = make_trace_with_turns()
        evaluator = Evaluator(metrics=[BrokenMetric])
        report = evaluator.run(trace)

    assert len(report.results) == 1
    assert report.results[0].score.label == "error"


def test_evaluator_embed_trace(patched_llm):
    trace = make_trace_with_turns()
    evaluator = Evaluator(embed_trace=True)
    report = evaluator.run(trace)
    assert report.trace is not None
    assert report.trace.trace_id == trace.trace_id


def test_score_to_label():
    assert _score_to_label(0.9) == "excellent"
    assert _score_to_label(0.75) == "good"
    assert _score_to_label(0.6) == "fair"
    assert _score_to_label(0.3) == "poor"
