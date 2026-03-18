"""
Multi-Turn Coherence Metric (Outcome — LLM Judge)

Evaluates whether the agent maintains context and coherence across the full conversation:
- Remembers information shared earlier in the call
- Correct pronoun/reference resolution ("that", "it", "the one you mentioned")
- Consistent topic tracking — doesn't contradict itself
- Doesn't ask for information the user already provided
"""

from __future__ import annotations

from voice_agent_evals.metrics.base import (
    BaseMetric,
    MetricResult,
    MetricScore,
    call_llm_judge,
    parse_score_response,
)
from voice_agent_evals.trace import VoiceTrace

SYSTEM_PROMPT = """You are an expert evaluator of voice AI agents. Your job is to assess
multi-turn coherence — whether the agent maintains context and consistency across
the full conversation.

Evaluate the conversation on a scale of 0.0 to 1.0 based on these criteria:

**High coherence (high score)**:
- Agent correctly recalls information mentioned earlier in the call
- Correct reference resolution (pronouns, "the thing you mentioned", etc.)
- No contradictions between earlier and later turns
- Agent does NOT ask for information the user already provided
- Topic transitions are smooth and tracked correctly

**Low coherence (low score)**:
- Agent forgets what the user said 2+ turns ago
- Agent asks for information already provided ("What was your name again?")
- Agent contradicts what it said in an earlier turn
- Agent loses track of the current topic
- Broken reference resolution ("Which account do you mean?" when only one was mentioned)

Respond with JSON only:
{
  "score": <float 0.0-1.0>,
  "label": <"excellent" | "good" | "fair" | "poor">,
  "reasoning": "<2-3 sentence explanation>",
  "coherence_failures": ["<specific failure if any>"],
  "context_tracked": ["<information successfully tracked across turns>"]
}"""


def _build_prompt(trace: VoiceTrace) -> str:
    lines = [
        f"Conversation has {len(trace.turns)} turns.\n",
        "Full transcript:\n",
    ]
    for i, turn in enumerate(trace.turns):
        label = turn.speaker.value.upper()
        text = turn.transcript or "[no transcript]"
        lines.append(f"[Turn {i+1}] {label}: {text}")
    return "\n".join(lines)


class CoherenceMetric(BaseMetric):
    name = "coherence"

    def evaluate(self, trace: VoiceTrace) -> MetricResult:
        if len(trace.turns) < 2:
            return MetricResult(
                metric_name=self.name,
                trace_id=trace.trace_id,
                score=MetricScore(
                    score=1.0,
                    label="not_applicable",
                    reasoning="Fewer than 2 turns — coherence is not meaningful.",
                ),
            )

        prompt = _build_prompt(trace)
        raw = call_llm_judge(SYSTEM_PROMPT, prompt)

        try:
            parsed = parse_score_response(raw)
            score = MetricScore(
                score=float(parsed["score"]),
                label=parsed.get("label", "unknown"),
                reasoning=parsed.get("reasoning", ""),
                details={
                    "coherence_failures": parsed.get("coherence_failures", []),
                    "context_tracked": parsed.get("context_tracked", []),
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
