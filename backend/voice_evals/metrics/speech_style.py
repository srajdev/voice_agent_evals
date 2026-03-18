"""
Speech Style Metrics (Tier 3 — LLM-judged + Algorithmic)

Three metrics in one module:

1. **VerbosityMatchMetric** — checks that agent response lengths are proportional to user turns.
   Algorithmic ratio-based pass + LLM contextual review.
   Score = blend of ratio-based score + LLM judgment.

2. **EmpathyMetric** — LLM identifies emotionally-loaded user turns and scores whether the
   agent acknowledged the user's emotional state appropriately.

3. **VocabularyMatchMetric** — LLM compares formality/complexity of agent language vs. user
   language across the conversation, assessing whether they match in register and lexical level.
"""

from __future__ import annotations

import statistics
from typing import Any

from voice_evals.metrics.base import (
    BaseMetric,
    MetricResult,
    MetricScore,
    call_llm_judge,
    parse_score_response,
)
from voice_evals.trace import VoiceTrace

# ──────────────────────────────────────────────────────────────────────────────
# Verbosity Match
# ──────────────────────────────────────────────────────────────────────────────

_VERBOSITY_SYSTEM_PROMPT = """You are an expert evaluator of voice AI agents.
Assess whether the agent's response lengths are contextually appropriate given the user's turns.

You will receive a transcript and per-turn verbosity ratios (agent_words / user_words).
Ideal ratio: 0.5–2.5x. Ratios >3x suggest the agent is over-explaining; <0.3x may be too terse.

Consider context: some situations warrant longer explanations (complex questions, errors)
or very short replies (simple confirmations). Apply good judgment.

Respond with JSON only:
{
  "score": <float 0.0-1.0>,
  "label": <"excellent"|"good"|"fair"|"poor">,
  "reasoning": "<2-3 sentence explanation>",
  "verbose_turns": ["<description of overly verbose turn>"],
  "terse_turns": ["<description of overly terse turn>"]
}"""


def _word_count(text: str | None) -> int:
    return len((text or "").split())


def _compute_verbosity_ratios(trace: VoiceTrace) -> list[dict[str, Any]]:
    pairs = trace.get_turn_pairs()
    ratios = []
    for user_turn, agent_turn in pairs:
        user_words = _word_count(user_turn.transcript)
        agent_words = _word_count(agent_turn.transcript)
        ratio = agent_words / user_words if user_words > 0 else None
        ratios.append({
            "user_text": user_turn.transcript or "",
            "agent_text": agent_turn.transcript or "",
            "user_words": user_words,
            "agent_words": agent_words,
            "ratio": ratio,
        })
    return ratios


def _ratio_based_score(ratios: list[dict[str, Any]]) -> float:
    """Score based purely on ratio thresholds."""
    if not ratios:
        return 1.0
    valid = [r for r in ratios if r["ratio"] is not None]
    if not valid:
        return 1.0
    good_count = sum(1 for r in valid if 0.3 <= r["ratio"] <= 3.0)
    return good_count / len(valid)


def _build_verbosity_prompt(trace: VoiceTrace, ratios: list[dict[str, Any]]) -> str:
    lines = ["Conversation transcript with verbosity ratios:\n"]
    for i, pair in enumerate(ratios):
        ratio_str = f"{pair['ratio']:.2f}x" if pair["ratio"] is not None else "N/A"
        lines.append(f"Turn pair {i + 1}:")
        lines.append(f"  USER ({pair['user_words']} words): {pair['user_text']}")
        lines.append(
            f"  AGENT ({pair['agent_words']} words, ratio={ratio_str}): {pair['agent_text']}"
        )
        lines.append("")
    return "\n".join(lines)


class VerbosityMatchMetric(BaseMetric):
    name = "verbosity_match"

    def evaluate(self, trace: VoiceTrace) -> MetricResult:
        pairs = trace.get_turn_pairs()
        if not pairs:
            return MetricResult(
                metric_name=self.name,
                trace_id=trace.trace_id,
                score=MetricScore(
                    score=0.0,
                    label="no_data",
                    reasoning="No user→agent turn pairs found.",
                    details={},
                ),
            )

        ratios = _compute_verbosity_ratios(trace)
        algo_score = _ratio_based_score(ratios)

        prompt = _build_verbosity_prompt(trace, ratios)
        raw = call_llm_judge(_VERBOSITY_SYSTEM_PROMPT, prompt)

        try:
            parsed = parse_score_response(raw)
            llm_score = float(parsed.get("score", algo_score))
            label = parsed.get("label", "unknown")
            reasoning = parsed.get("reasoning", "")
            verbose_turns = parsed.get("verbose_turns", [])
            terse_turns = parsed.get("terse_turns", [])
        except Exception as e:
            return MetricResult(
                metric_name=self.name,
                trace_id=trace.trace_id,
                score=MetricScore(
                    score=0.0,
                    label="error",
                    reasoning=f"Failed to parse LLM response: {e}",
                    details={"raw": raw},
                ),
                raw_response=raw,
            )

        # Blend algorithmic + LLM scores (equal weight)
        final_score = (algo_score + llm_score) / 2
        final_score = max(0.0, min(1.0, final_score))

        valid_ratios = [r["ratio"] for r in ratios if r["ratio"] is not None]
        details: dict[str, Any] = {
            "n_pairs": len(pairs),
            "mean_ratio": round(statistics.mean(valid_ratios), 2) if valid_ratios else None,
            "algo_score": round(algo_score, 3),
            "llm_score": round(llm_score, 3),
            "verbose_turns": verbose_turns,
            "terse_turns": terse_turns,
        }

        return MetricResult(
            metric_name=self.name,
            trace_id=trace.trace_id,
            score=MetricScore(
                score=final_score, label=label, reasoning=reasoning, details=details
            ),
            raw_response=raw,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Empathy
# ──────────────────────────────────────────────────────────────────────────────

_EMPATHY_SYSTEM_PROMPT = """You are an expert evaluator of voice AI agent conversations,
specialising in emotional intelligence and empathy.

Your job is to:
1. Identify user turns that express frustration, distress, confusion, excitement, or other
   emotionally meaningful states.
2. Evaluate whether the agent acknowledged, validated, or appropriately responded to those
   emotional signals.

Score from 0.0 to 1.0:
- 1.0: Agent consistently and naturally acknowledges user emotions
- 0.75: Agent usually acknowledges emotions but misses some signals
- 0.5: Agent occasionally shows empathy but often responds robotically
- 0.25: Agent rarely acknowledges emotional state
- 0.0: Agent completely ignores emotional signals

If no emotionally meaningful user turns exist, score 1.0 (not applicable).

Respond with JSON only:
{
  "score": <float 0.0-1.0>,
  "label": <"excellent"|"good"|"fair"|"poor"|"not_applicable">,
  "reasoning": "<2-3 sentence explanation>",
  "emotional_moments": [
    {"user_text": "<quote>", "emotion": "<emotion type>",
     "agent_response": "<quote>", "handled_well": <bool>}
  ]
}"""


class EmpathyMetric(BaseMetric):
    name = "empathy"

    def evaluate(self, trace: VoiceTrace) -> MetricResult:
        if not trace.agent_turns:
            return MetricResult(
                metric_name=self.name,
                trace_id=trace.trace_id,
                score=MetricScore(
                    score=0.0,
                    label="no_data",
                    reasoning="No agent turns found in the trace.",
                    details={},
                ),
            )

        prompt = _build_full_transcript_prompt(trace)
        raw = call_llm_judge(_EMPATHY_SYSTEM_PROMPT, prompt)

        try:
            parsed = parse_score_response(raw)
            score = float(parsed.get("score", 0.0))
            label = parsed.get("label", "unknown")
            reasoning = parsed.get("reasoning", "")
            moments = parsed.get("emotional_moments", [])
        except Exception as e:
            return MetricResult(
                metric_name=self.name,
                trace_id=trace.trace_id,
                score=MetricScore(
                    score=0.0,
                    label="error",
                    reasoning=f"Failed to parse LLM response: {e}",
                    details={"raw": raw},
                ),
                raw_response=raw,
            )

        return MetricResult(
            metric_name=self.name,
            trace_id=trace.trace_id,
            score=MetricScore(
                score=score,
                label=label,
                reasoning=reasoning,
                details={"emotional_moments": moments},
            ),
            raw_response=raw,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary Match
# ──────────────────────────────────────────────────────────────────────────────

_VOCABULARY_SYSTEM_PROMPT = """You are an expert evaluator of voice AI agent conversations,
specialising in language register and lexical matching.

Assess whether the agent's word choice, formality level, and vocabulary complexity
appropriately mirror the user's communication style throughout the conversation.

Good vocabulary match means:
- If the user speaks casually, the agent also uses casual, everyday language
- If the user is formal or technical, the agent matches that register
- The agent avoids jargon the user hasn't introduced
- The agent doesn't "talk down" or "talk up" to the user

Score from 0.0 to 1.0:
- 1.0: Agent perfectly mirrors the user's register and vocabulary throughout
- 0.75: Mostly matches but occasionally misses cues
- 0.5: Noticeable mismatches in formality or complexity
- 0.25: Agent consistently uses an inappropriate register
- 0.0: Complete vocabulary mismatch throughout

Respond with JSON only:
{
  "score": <float 0.0-1.0>,
  "label": <"excellent"|"good"|"fair"|"poor">,
  "reasoning": "<2-3 sentence explanation>",
  "user_register": "<description of user's language style>",
  "agent_register": "<description of agent's language style>",
  "mismatch_examples": ["<example of mismatch>"]
}"""


class VocabularyMatchMetric(BaseMetric):
    name = "vocabulary_match"

    def evaluate(self, trace: VoiceTrace) -> MetricResult:
        if not trace.agent_turns or not trace.user_turns:
            return MetricResult(
                metric_name=self.name,
                trace_id=trace.trace_id,
                score=MetricScore(
                    score=0.0,
                    label="no_data",
                    reasoning="Insufficient turns to assess vocabulary match.",
                    details={},
                ),
            )

        prompt = _build_full_transcript_prompt(trace)
        raw = call_llm_judge(_VOCABULARY_SYSTEM_PROMPT, prompt)

        try:
            parsed = parse_score_response(raw)
            score = float(parsed.get("score", 0.0))
            label = parsed.get("label", "unknown")
            reasoning = parsed.get("reasoning", "")
        except Exception as e:
            return MetricResult(
                metric_name=self.name,
                trace_id=trace.trace_id,
                score=MetricScore(
                    score=0.0,
                    label="error",
                    reasoning=f"Failed to parse LLM response: {e}",
                    details={"raw": raw},
                ),
                raw_response=raw,
            )

        details: dict[str, Any] = {
            "user_register": parsed.get("user_register", ""),
            "agent_register": parsed.get("agent_register", ""),
            "mismatch_examples": parsed.get("mismatch_examples", []),
        }

        return MetricResult(
            metric_name=self.name,
            trace_id=trace.trace_id,
            score=MetricScore(score=score, label=label, reasoning=reasoning, details=details),
            raw_response=raw,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_full_transcript_prompt(trace: VoiceTrace) -> str:
    lines = ["Full conversation transcript:\n"]
    for turn in trace.turns:
        label = turn.speaker.value.upper()
        text = turn.transcript or "[no transcript]"
        lines.append(f"{label}: {text}")
    return "\n".join(lines)
