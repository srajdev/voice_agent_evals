"""
Base types and LLM client for all metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from voice_evals.trace import VoiceTrace


class MetricScore(BaseModel):
    """Numeric score with explanation."""

    score: float          # 0.0–1.0 (or 0–100 for some metrics — always normalized before display)
    label: str            # e.g. "good", "needs_improvement", "poor"
    reasoning: str        # LLM's explanation of why this score was given
    details: dict[str, Any] = {}  # metric-specific breakdown


class MetricResult(BaseModel):
    """Full result from running one metric against a VoiceTrace."""

    metric_name: str
    trace_id: str
    score: MetricScore
    raw_response: str | None = None  # raw LLM output for debugging


class BaseMetric(ABC):
    """Abstract base for all evaluation metrics."""

    name: str

    @abstractmethod
    def evaluate(self, trace: VoiceTrace) -> MetricResult:
        """Run this metric against a VoiceTrace and return a result."""


def get_llm_client():
    """Return an Anthropic client. Reads ANTHROPIC_API_KEY from environment."""
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic") from e

    return anthropic.Anthropic()


def call_llm_judge(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    max_tokens: int = 1024,
) -> str:
    """
    Call Claude as an LLM judge. Returns raw text response.

    Uses structured output prompting — caller is responsible for parsing JSON.
    """
    import os
    resolved_model = model or os.environ.get("VOICE_EVALS_MODEL", "claude-sonnet-4-6")
    client = get_llm_client()
    message = client.messages.create(
        model=resolved_model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


def parse_score_response(raw: str) -> dict[str, Any]:
    """
    Parse a JSON score response from the LLM judge.

    The LLM is prompted to respond with JSON; this handles common formatting issues.
    """
    import json
    import re

    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    # Find the first JSON object
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    return json.loads(cleaned)
