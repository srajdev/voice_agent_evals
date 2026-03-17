"""
VoiceTrace — canonical, platform-agnostic representation of a voice conversation.

Think OpenTelemetry spans, but for voice. Two creation paths:
  1. Upload mode: audio file → VoiceTrace
  2. Live mode: platform webhooks → VoiceTrace (built incrementally)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Speaker(str, Enum):
    USER = "user"
    AGENT = "agent"
    UNKNOWN = "unknown"


class AudioSegment(BaseModel):
    """Reference to a slice of audio for a specific turn."""

    file_path: str | None = None
    start_ms: float | None = None  # offset from recording start
    end_ms: float | None = None
    channel: int | None = None  # 0 = left, 1 = right (stereo), None = mono/mixed


class TimingInfo(BaseModel):
    """Precise timing boundaries for a conversation turn."""

    speech_start_ms: float | None = None  # when speaker started talking
    speech_end_ms: float | None = None  # when speaker stopped talking
    # For agent turns: time from user speech_end to first audio byte
    ttfw_ms: float | None = None
    # Source of timing data
    source: str = "estimated"  # "platform" | "vad" | "estimated"


class Turn(BaseModel):
    """A single conversational turn — one speaker's contribution."""

    turn_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    speaker: Speaker
    transcript: str | None = None
    transcript_confidence: float | None = None  # 0.0–1.0
    audio: AudioSegment | None = None
    timing: TimingInfo | None = None
    # Optional: expected intent label from scenario config
    expected_intent: str | None = None
    # Optional: metadata from platform (e.g. Vapi turn metadata)
    platform_metadata: dict[str, Any] = Field(default_factory=dict)


class PlatformInfo(BaseModel):
    """Where this trace came from."""

    platform: str = "upload"  # "upload" | "vapi" | "retell" | "twilio" | "bland" | "generic"
    call_id: str | None = None
    agent_id: str | None = None
    phone_number_from: str | None = None
    phone_number_to: str | None = None
    raw_events: list[dict[str, Any]] = Field(default_factory=list)


class AudioInfo(BaseModel):
    """Metadata about the source audio file(s)."""

    original_file: str | None = None
    duration_ms: float | None = None
    sample_rate: int | None = None
    channels: int | None = None  # 1 = mono, 2 = stereo
    format: str | None = None  # "wav" | "mp3" | "ogg" etc.
    # For stereo: which channel is which
    user_channel: int | None = None   # e.g. 0 (left)
    agent_channel: int | None = None  # e.g. 1 (right)


class ScenarioConfig(BaseModel):
    """Optional test scenario — what *should* have happened."""

    scenario_id: str | None = None
    description: str | None = None
    # What task was the user trying to accomplish?
    expected_task: str | None = None
    # Expected intents per turn (list aligned with turn order, user turns only)
    expected_intents: list[str] = Field(default_factory=list)
    # What constitutes task completion?
    completion_criteria: str | None = None
    # User persona used in the test
    user_persona: str | None = None


class VoiceTrace(BaseModel):
    """
    Canonical representation of a voice conversation.

    Platform-agnostic — can be created from an uploaded audio file,
    a live webhook stream, or constructed manually.
    """

    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Core conversation data
    turns: list[Turn] = Field(default_factory=list)

    # Source audio metadata
    audio_info: AudioInfo | None = None

    # Platform / origin metadata
    platform_info: PlatformInfo = Field(default_factory=PlatformInfo)

    # Optional: test scenario this trace was evaluated against
    scenario: ScenarioConfig | None = None

    # Call-level timing
    call_start_at: datetime | None = None
    call_end_at: datetime | None = None

    # Free-form trace-level metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #

    @property
    def user_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.speaker == Speaker.USER]

    @property
    def agent_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.speaker == Speaker.AGENT]

    @property
    def duration_ms(self) -> float | None:
        if self.audio_info and self.audio_info.duration_ms:
            return self.audio_info.duration_ms
        if self.call_start_at and self.call_end_at:
            delta = self.call_end_at - self.call_start_at
            return delta.total_seconds() * 1000
        return None

    @property
    def full_transcript(self) -> str:
        """Interleaved transcript of all turns."""
        lines = []
        for turn in self.turns:
            speaker_label = turn.speaker.value.upper()
            text = turn.transcript or "[no transcript]"
            lines.append(f"{speaker_label}: {text}")
        return "\n".join(lines)

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)

    def get_turn_pairs(self) -> list[tuple[Turn, Turn]]:
        """Return (user_turn, agent_turn) pairs for Q&A-style analysis."""
        pairs: list[tuple[Turn, Turn]] = []
        i = 0
        while i < len(self.turns) - 1:
            if self.turns[i].speaker == Speaker.USER and self.turns[i + 1].speaker == Speaker.AGENT:
                pairs.append((self.turns[i], self.turns[i + 1]))
                i += 2
            else:
                i += 1
        return pairs
