"""
Conversation Quality Metric (Tier 1 — LLM Judge)

Evaluates whether the voice agent follows good voice-first communication principles:
- Short, natural sentences (not reading a bulleted list)
- No filler-heavy or over-qualified language
- Appropriate pacing cues and natural pauses
- No robotic, stiff, or overly formal phrasing
- Appropriate response length for voice (not wall-of-text responses)
"""

from __future__ import annotations

from voice_evals.metrics.base import (
    BaseMetric,
    MetricResult,
    MetricScore,
    call_llm_judge,
    parse_score_response,
)
from voice_evals.trace import Speaker, VoiceTrace

SYSTEM_PROMPT = """You are an expert evaluator of voice AI agents. Your job is to assess whether
the agent's speech follows good voice-first communication principles.

You will be given a transcript of a voice conversation. Evaluate ONLY the agent's turns.

Score the conversation on a scale of 0.0 to 1.0 based on these criteria:

**Good voice communication (high score)**:
- Short, conversational sentences (typically 1-2 sentences per turn)
- Natural language — sounds like someone talking, not writing
- No bullet points or numbered lists read aloud
- Appropriate use of affirmations ("Got it", "Sure", "Of course")
- Clear and direct — no over-qualification
- Appropriate length — doesn't ramble or repeat itself

**Poor voice communication (low score)**:
- Long run-on sentences or walls of text
- Reading out structured lists (e.g., "Option 1..., Option 2..., Option 3...")
- Robotic or overly formal phrasing ("Certainly! I would be happy to assist you with that.")
- Excessive filler or hedge words ("Actually", "Basically", "It's important to note that")
- Repeating information that was just said
- Very long responses when a short one was warranted

Respond with JSON only, in this exact format:
{
  "score": <float 0.0-1.0>,
  "label": <"excellent" | "good" | "fair" | "poor">,
  "reasoning": "<2-3 sentence explanation>",
  "issues": ["<specific issue 1>", "<specific issue 2>"],
  "strengths": ["<strength 1>", "<strength 2>"]
}"""


def _build_prompt(trace: VoiceTrace) -> str:
    lines = ["Conversation transcript:\n"]
    for turn in trace.turns:
        label = turn.speaker.value.upper()
        text = turn.transcript or "[no transcript]"
        lines.append(f"{label}: {text}")
    return "\n".join(lines)


class ConversationQualityMetric(BaseMetric):
    name = "conversation_quality"

    def evaluate(self, trace: VoiceTrace) -> MetricResult:
        agent_turns = trace.agent_turns
        if not agent_turns:
            return MetricResult(
                metric_name=self.name,
                trace_id=trace.trace_id,
                score=MetricScore(
                    score=0.0,
                    label="no_data",
                    reasoning="No agent turns found in the trace.",
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
                    "issues": parsed.get("issues", []),
                    "strengths": parsed.get("strengths", []),
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
