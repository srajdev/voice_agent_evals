"""
Intent Accuracy Metric (Tier 1 — LLM Judge)

Evaluates whether the agent correctly interpreted user intent for each turn.
Requires a ScenarioConfig with expected_intents populated.

If no expected intents are provided, falls back to inferred intent assessment
(LLM estimates what the user likely intended and whether the agent honored it).
"""

from __future__ import annotations

from voice_agent_evals.metrics.base import (
    BaseMetric,
    MetricResult,
    MetricScore,
    call_llm_judge,
    parse_score_response,
)
from voice_agent_evals.trace import Speaker, Turn, VoiceTrace

SYSTEM_PROMPT_WITH_EXPECTED = """You are an expert evaluator of voice AI agents. Your job is to
assess whether the agent correctly interpreted the user's intent at each turn.

You will be given a list of conversation turns with the EXPECTED intent label for each user turn,
and the agent's response. Determine whether the agent's response demonstrates correct intent
understanding.

Score 0.0–1.0 overall (average across turns, with more weight on incorrect turns).

Respond with JSON only:
{
  "score": <float 0.0-1.0>,
  "label": <"excellent" | "good" | "fair" | "poor">,
  "reasoning": "<2-3 sentence summary>",
  "turn_scores": [
    {
      "turn_index": <int>,
      "expected_intent": "<intent label>",
      "inferred_intent": "<what the agent seemed to interpret>",
      "correct": <true|false>,
      "note": "<optional explanation>"
    }
  ]
}"""

SYSTEM_PROMPT_INFERRED = """You are an expert evaluator of voice AI agents. Your job is to
assess whether the agent correctly interpreted what the user was asking at each turn.

For each user turn, infer the most likely user intent, then assess whether the agent's
response addresses that intent.

Score 0.0–1.0 overall.

Respond with JSON only:
{
  "score": <float 0.0-1.0>,
  "label": <"excellent" | "good" | "fair" | "poor">,
  "reasoning": "<2-3 sentence summary>",
  "turn_scores": [
    {
      "turn_index": <int>,
      "inferred_intent": "<what the user was asking>",
      "agent_addressed": <true|false>,
      "note": "<optional>"
    }
  ]
}"""


def _build_prompt_with_expected(
    turns: list[Turn],
    expected_intents: list[str],
) -> str:
    lines = ["Turn-by-turn analysis:\n"]
    intent_idx = 0
    for i, turn in enumerate(turns):
        label = turn.speaker.value.upper()
        text = turn.transcript or "[no transcript]"

        if turn.speaker == Speaker.USER:
            intent = (
                expected_intents[intent_idx]
                if intent_idx < len(expected_intents)
                else turn.expected_intent or "unknown"
            )
            lines.append(f"[Turn {i+1}] {label} (expected_intent={intent!r}): {text}")
            intent_idx += 1
        else:
            lines.append(f"[Turn {i+1}] {label}: {text}")

    return "\n".join(lines)


def _build_prompt_inferred(turns: list[Turn]) -> str:
    lines = ["Conversation:\n"]
    for i, turn in enumerate(turns):
        label = turn.speaker.value.upper()
        text = turn.transcript or "[no transcript]"
        lines.append(f"[Turn {i+1}] {label}: {text}")
    return "\n".join(lines)


class IntentAccuracyMetric(BaseMetric):
    name = "intent_accuracy"

    def evaluate(self, trace: VoiceTrace) -> MetricResult:
        if not trace.turns:
            return MetricResult(
                metric_name=self.name,
                trace_id=trace.trace_id,
                score=MetricScore(
                    score=0.0,
                    label="no_data",
                    reasoning="No turns in trace.",
                ),
            )

        has_expected = (
            trace.scenario is not None
            and len(trace.scenario.expected_intents) > 0
        ) or any(t.expected_intent for t in trace.turns)

        if has_expected:
            expected_intents = (
                trace.scenario.expected_intents if trace.scenario else []
            )
            prompt = _build_prompt_with_expected(trace.turns, expected_intents)
            raw = call_llm_judge(SYSTEM_PROMPT_WITH_EXPECTED, prompt)
        else:
            prompt = _build_prompt_inferred(trace.turns)
            raw = call_llm_judge(SYSTEM_PROMPT_INFERRED, prompt)

        try:
            parsed = parse_score_response(raw)
            score = MetricScore(
                score=float(parsed["score"]),
                label=parsed.get("label", "unknown"),
                reasoning=parsed.get("reasoning", ""),
                details={
                    "mode": "expected" if has_expected else "inferred",
                    "turn_scores": parsed.get("turn_scores", []),
                },
            )
        except Exception as e:
            score = MetricScore(
                score=0.0,
                label="error",
                reasoning=f"Failed to parse LLM response: {e}",
                details={"raw": raw},
            )

        return MetricResult(
            metric_name=self.name,
            trace_id=trace.trace_id,
            score=score,
            raw_response=raw,
        )
