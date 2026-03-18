"""
VAD Quality Metrics (Tier 2)

Two metrics in one module:

1. **VadQualityMetric** — detects VAD false positives (background noise triggering unexpected
   agent responses). Heuristic: flags agent turns with <5 words AND low transcript_confidence,
   then uses an LLM to confirm whether each flagged turn looks noise-triggered.
   Score = 1 - (confirmed_false_positives / total_agent_turns)

2. **InterruptionRecoveryMetric** — evaluates how well the agent handles user interruptions.
   Detects overlaps via `is_overlap` in platform_metadata or negative inter-turn latency.
   LLM scores each interruption event 0/0.5/1.0 (missed/partial/recovered).
   Returns `not_applicable` when no overlaps exist.
"""

from __future__ import annotations

from typing import Any

from voice_evals.metrics.base import (
    BaseMetric,
    MetricResult,
    MetricScore,
    call_llm_judge,
    parse_score_response,
)
from voice_evals.trace import Speaker, Turn, VoiceTrace

# ──────────────────────────────────────────────────────────────────────────────
# VAD False Positive Detection
# ──────────────────────────────────────────────────────────────────────────────

_VAD_SYSTEM_PROMPT = """You are an expert at detecting voice activity detection (VAD) errors in
transcribed conversations. VAD false positives occur when background noise or silence is mistakenly
interpreted as speech, resulting in garbled, empty, or context-free agent utterances.

You will receive a list of short, low-confidence agent turns along with surrounding context.
For each flagged turn, determine whether it is likely a VAD false positive (noise artifact) or a
legitimate (if brief) agent response.

Respond with JSON only:
{
  "confirmed_false_positives": <int — count of turns confirmed as false positives>,
  "turn_verdicts": [
    {"turn_index": <int>, "is_false_positive": <bool>, "reason": "<brief reason>"}
  ],
  "reasoning": "<overall assessment>"
}"""


def _word_count(text: str | None) -> int:
    return len((text or "").split())


def _flag_suspicious_turns(trace: VoiceTrace) -> list[dict[str, Any]]:
    """Return agent turns that are heuristically suspicious (short + low confidence)."""
    flagged = []
    for i, turn in enumerate(trace.turns):
        if turn.speaker != Speaker.AGENT:
            continue
        words = _word_count(turn.transcript)
        conf = turn.transcript_confidence
        if words < 5 and (conf is None or conf < 0.6):
            flagged.append({"index": i, "turn": turn})
    return flagged


def _build_vad_prompt(trace: VoiceTrace, flagged: list[dict[str, Any]]) -> str:
    lines = ["Flagged agent turns (short + low confidence):\n"]
    for item in flagged:
        idx = item["index"]
        turn = item["turn"]
        conf_val = turn.transcript_confidence
        conf = f"{conf_val:.2f}" if conf_val is not None else "N/A"
        lines.append(f"Turn {idx} [confidence={conf}]: \"{turn.transcript or ''}\"")

        # Add context window (±2 turns)
        context_start = max(0, idx - 2)
        context_end = min(len(trace.turns) - 1, idx + 2)
        lines.append("  Context:")
        for j in range(context_start, context_end + 1):
            t = trace.turns[j]
            marker = " <<< FLAGGED" if j == idx else ""
            lines.append(f"    [{t.speaker.value.upper()}]: {t.transcript or ''}{marker}")
        lines.append("")

    return "\n".join(lines)


class VadQualityMetric(BaseMetric):
    name = "vad_quality"

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
                    details={},
                ),
            )

        flagged = _flag_suspicious_turns(trace)
        total = len(agent_turns)
        details: dict[str, Any] = {
            "total_agent_turns": total,
            "heuristic_flags": len(flagged),
        }

        if not flagged:
            return MetricResult(
                metric_name=self.name,
                trace_id=trace.trace_id,
                score=MetricScore(
                    score=1.0,
                    label="excellent",
                    reasoning="No suspicious agent turns detected; VAD quality appears clean.",
                    details=details,
                ),
            )

        prompt = _build_vad_prompt(trace, flagged)
        raw = call_llm_judge(_VAD_SYSTEM_PROMPT, prompt)

        try:
            parsed = parse_score_response(raw)
            confirmed = int(parsed.get("confirmed_false_positives", 0))
            verdicts = parsed.get("turn_verdicts", [])
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

        score = 1.0 - (confirmed / total)
        score = max(0.0, min(1.0, score))
        label = _score_label(score)

        details.update({
            "confirmed_false_positives": confirmed,
            "turn_verdicts": verdicts,
        })

        return MetricResult(
            metric_name=self.name,
            trace_id=trace.trace_id,
            score=MetricScore(score=score, label=label, reasoning=reasoning, details=details),
            raw_response=raw,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Interruption Recovery
# ──────────────────────────────────────────────────────────────────────────────

_INTERRUPTION_SYSTEM_PROMPT = """You are an expert evaluator of voice AI agent conversations.
Your job is to assess how well the agent handles user interruptions.

You will receive a list of interruption events, each showing:
- The user turn that interrupted the agent
- The agent's next response after the interruption

Score each interruption on a 3-point scale:
- 1.0 = Fully recovered: agent acknowledged or correctly addressed the interruption
- 0.5 = Partial recovery: agent continued but showed some awareness of the interruption
- 0.0 = Missed: agent ignored the interruption and continued as if it hadn't happened

Respond with JSON only:
{
  "interruption_scores": [
    {"event_index": <int>, "score": <0.0|0.5|1.0>, "reason": "<brief reason>"}
  ],
  "overall_score": <float 0.0-1.0>,
  "label": <"excellent"|"good"|"fair"|"poor">,
  "reasoning": "<2-3 sentence overall assessment>"
}"""


def _find_interruption_events(trace: VoiceTrace) -> list[dict[str, Any]]:
    """Find turns flagged as overlaps or with negative inter-turn latency."""
    events = []
    turns = trace.turns

    for i, turn in enumerate(turns):
        is_interruption = False

        # Check platform_metadata flag (set by WhisperX diarization)
        if turn.platform_metadata.get("is_overlap"):
            is_interruption = True

        # Check negative latency: user started speaking before agent finished
        if (
            not is_interruption
            and i > 0
            and turn.speaker == Speaker.USER
            and turns[i - 1].speaker == Speaker.AGENT
        ):
            prev = turns[i - 1]
            if (
                turn.timing
                and turn.timing.speech_start_ms is not None
                and prev.timing
                and prev.timing.speech_end_ms is not None
                and turn.timing.speech_start_ms < prev.timing.speech_end_ms
            ):
                is_interruption = True

        if not is_interruption:
            continue

        # Find the next agent response after this turn
        next_agent: Turn | None = None
        for j in range(i + 1, len(turns)):
            if turns[j].speaker == Speaker.AGENT:
                next_agent = turns[j]
                break

        events.append({
            "event_index": len(events),
            "interrupting_turn": turn,
            "next_agent_turn": next_agent,
        })

    return events


def _build_interruption_prompt(events: list[dict[str, Any]]) -> str:
    lines = [f"Found {len(events)} interruption event(s):\n"]
    for ev in events:
        idx = ev["event_index"]
        user_turn: Turn = ev["interrupting_turn"]
        agent_turn: Turn | None = ev["next_agent_turn"]
        lines.append(f"Event {idx}:")
        lines.append(f"  USER (interrupted): {user_turn.transcript or '[no transcript]'}")
        agent_text = agent_turn.transcript if agent_turn else "[no agent response followed]"
        lines.append(f"  AGENT (next response): {agent_text}")
        lines.append("")
    return "\n".join(lines)


class InterruptionRecoveryMetric(BaseMetric):
    name = "interruption_recovery"

    def evaluate(self, trace: VoiceTrace) -> MetricResult:
        events = _find_interruption_events(trace)

        if not events:
            return MetricResult(
                metric_name=self.name,
                trace_id=trace.trace_id,
                score=MetricScore(
                    score=1.0,
                    label="not_applicable",
                    reasoning="No interruption events detected in this conversation.",
                    details={"n_interruptions": 0},
                ),
            )

        prompt = _build_interruption_prompt(events)
        raw = call_llm_judge(_INTERRUPTION_SYSTEM_PROMPT, prompt)

        try:
            parsed = parse_score_response(raw)
            overall_score = float(parsed.get("overall_score", 0.0))
            label = parsed.get("label", "unknown")
            reasoning = parsed.get("reasoning", "")
            interruption_scores = parsed.get("interruption_scores", [])
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
            "n_interruptions": len(events),
            "interruption_scores": interruption_scores,
        }

        return MetricResult(
            metric_name=self.name,
            trace_id=trace.trace_id,
            score=MetricScore(
                score=overall_score, label=label, reasoning=reasoning, details=details
            ),
            raw_response=raw,
        )


def _score_label(score: float) -> str:
    if score >= 0.9:
        return "excellent"
    if score >= 0.7:
        return "good"
    if score >= 0.5:
        return "fair"
    return "poor"
