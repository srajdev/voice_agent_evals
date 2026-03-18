"""
Task Completion Metric (Outcome — LLM Judge)

Evaluates whether the scripted task was completed end-to-end.
- Binary: task completed yes/no
- Partial credit: % of task steps completed
- Requires ScenarioConfig.expected_task and optionally completion_criteria
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

SYSTEM_PROMPT = """You are an expert evaluator of voice AI agents. Your job is to determine
whether the agent successfully completed the user's task by the end of the conversation.

You will be given:
1. The task description — what the user was trying to accomplish
2. The completion criteria — what counts as success (if provided)
3. The full conversation transcript

Assess task completion on a scale of 0.0–1.0:
- 1.0 = task fully completed, user's goal achieved
- 0.5–0.9 = partially completed (main goal partially met, or with caveats)
- 0.1–0.4 = minimal progress, but task largely failed
- 0.0 = task not completed at all / wrong direction

Respond with JSON only:
{
  "score": <float 0.0-1.0>,
  "label": <"completed" | "partial" | "failed">,
  "reasoning": "<2-3 sentence explanation of what happened>",
  "completed_steps": ["<step that was done>"],
  "missing_steps": ["<step that was not done>"],
  "blockers": ["<what prevented completion, if anything>"]
}"""


def _build_prompt(trace: VoiceTrace) -> str:
    task = "Not specified"
    criteria = "Not specified"
    if trace.scenario:
        task = trace.scenario.expected_task or "Not specified"
        criteria = trace.scenario.completion_criteria or "Not specified"

    lines = [
        f"Task: {task}",
        f"Completion criteria: {criteria}",
        "",
        "Full conversation transcript:",
    ]
    for i, turn in enumerate(trace.turns):
        label = turn.speaker.value.upper()
        text = turn.transcript or "[no transcript]"
        lines.append(f"[Turn {i+1}] {label}: {text}")

    return "\n".join(lines)


class TaskCompletionMetric(BaseMetric):
    name = "task_completion"

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

        prompt = _build_prompt(trace)
        raw = call_llm_judge(SYSTEM_PROMPT, prompt)

        try:
            parsed = parse_score_response(raw)
            score = MetricScore(
                score=float(parsed["score"]),
                label=parsed.get("label", "unknown"),
                reasoning=parsed.get("reasoning", ""),
                details={
                    "completed_steps": parsed.get("completed_steps", []),
                    "missing_steps": parsed.get("missing_steps", []),
                    "blockers": parsed.get("blockers", []),
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
