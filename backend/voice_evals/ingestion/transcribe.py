"""
Transcription backend: convert audio → per-turn transcript.

Default backend: OpenAI Whisper (local, free).
Pluggable: implement TranscriptionBackend to add Deepgram, AssemblyAI, etc.

For stereo audio, transcribes each channel separately to get clean per-speaker turns.
For mono audio, uses Whisper's word-level timestamps to segment turns (best-effort).
"""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from voice_evals.trace import Speaker


@dataclass
class TranscribedSegment:
    """A single transcribed segment (maps to one Turn)."""

    text: str
    start_ms: float
    end_ms: float
    confidence: float | None = None
    words: list[dict] = field(default_factory=list)  # word-level timing if available


@dataclass
class TranscriptionResult:
    """Full transcription result for one audio channel/file."""

    segments: list[TranscribedSegment]
    language: str | None = None
    backend: str = "whisper"


class TranscriptionBackend(ABC):
    """Abstract base — implement this to add a new STT provider."""

    @abstractmethod
    def transcribe(self, audio_samples: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """Transcribe audio samples, return segmented result."""


class WhisperBackend(TranscriptionBackend):
    """
    Local Whisper transcription.

    Models: tiny, base, small, medium, large-v3
    Recommendation: "base" for dev/testing (fast), "medium" for production quality.
    """

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None  # lazy load

    def _get_model(self):
        if self._model is None:
            try:
                import whisper
            except ImportError as e:
                raise RuntimeError(
                    "Whisper not installed. Run: pip install openai-whisper"
                ) from e
            self._model = whisper.load_model(self.model_size)
        return self._model

    def transcribe(self, audio_samples: np.ndarray, sample_rate: int) -> TranscriptionResult:
        import whisper

        model = self._get_model()

        # Whisper expects float32 mono at 16kHz
        if audio_samples.ndim > 1:
            # Downmix to mono if needed (shouldn't happen if called per-channel)
            audio_samples = audio_samples.mean(axis=0)

        # Resample to 16kHz if needed (whisper internal requirement)
        if sample_rate != 16000:
            import librosa
            audio_samples = librosa.resample(audio_samples, orig_sr=sample_rate, target_sr=16000)

        # Write to temp WAV (Whisper's Python API accepts numpy arrays directly too)
        result = model.transcribe(
            audio_samples,
            word_timestamps=True,
            verbose=False,
        )

        segments = []
        for seg in result.get("segments", []):
            words = [
                {
                    "word": w.get("word", ""),
                    "start_ms": w.get("start", 0) * 1000,
                    "end_ms": w.get("end", 0) * 1000,
                    "probability": w.get("probability"),
                }
                for w in seg.get("words", [])
            ]
            # Average word probability as segment confidence
            probs = [w["probability"] for w in words if w.get("probability") is not None]
            confidence = float(np.mean(probs)) if probs else None

            segments.append(
                TranscribedSegment(
                    text=seg["text"].strip(),
                    start_ms=seg["start"] * 1000,
                    end_ms=seg["end"] * 1000,
                    confidence=confidence,
                    words=words,
                )
            )

        return TranscriptionResult(
            segments=segments,
            language=result.get("language"),
            backend="whisper",
        )


def transcribe_stereo(
    user_samples: np.ndarray,
    agent_samples: np.ndarray,
    sample_rate: int,
    backend: TranscriptionBackend | None = None,
) -> tuple[TranscriptionResult, TranscriptionResult]:
    """
    Transcribe stereo audio by transcribing each channel independently.

    Returns (user_result, agent_result).
    """
    if backend is None:
        backend = WhisperBackend()

    user_result = backend.transcribe(user_samples, sample_rate)
    agent_result = backend.transcribe(agent_samples, sample_rate)
    return user_result, agent_result


def transcribe_mono(
    samples: np.ndarray,
    sample_rate: int,
    backend: TranscriptionBackend | None = None,
) -> TranscriptionResult:
    """
    Transcribe mono audio — all speakers mixed together.

    Use this when stereo is unavailable. Speaker diarization is not implemented
    yet, so turns will need to be split by silence/VAD heuristics downstream.
    """
    if backend is None:
        backend = WhisperBackend()

    return backend.transcribe(samples, sample_rate)


def merge_and_sort_turns(
    user_result: TranscriptionResult,
    agent_result: TranscriptionResult,
) -> list[tuple[str, TranscribedSegment]]:
    """
    Interleave user and agent segments sorted by start time.

    Returns list of (speaker_label, segment) tuples.
    """
    combined: list[tuple[str, TranscribedSegment]] = []
    for seg in user_result.segments:
        combined.append(("user", seg))
    for seg in agent_result.segments:
        combined.append(("agent", seg))

    combined.sort(key=lambda x: x[1].start_ms)
    return combined
