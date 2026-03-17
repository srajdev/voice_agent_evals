"""
Transcription backend: convert audio → per-turn transcript.

Default backend: WhisperX (Whisper + pyannote diarization).
Pluggable: implement TranscriptionBackend to add Deepgram, AssemblyAI, etc.

For stereo audio, transcribes each channel separately — speaker is known from channel.
For mono audio, runs full diarization to attribute turns to user vs agent.

Requires:
    - pip install whisperx
    - HF_TOKEN env var for diarization (mono only)
      Accept license at: https://huggingface.co/pyannote/speaker-diarization-3.1
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TranscribedSegment:
    """A single transcribed segment (maps to one Turn)."""

    text: str
    start_ms: float
    end_ms: float
    confidence: float | None = None
    words: list[dict] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """Full transcription result for one audio channel/file."""

    segments: list[TranscribedSegment]
    language: str | None = None
    backend: str = "whisperx"


class TranscriptionBackend(ABC):
    """Abstract base — implement this to add a new STT provider."""

    @abstractmethod
    def transcribe(self, audio_samples: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """Transcribe audio samples, return segmented result."""


@dataclass
class DiarizedSegment:
    """A transcribed segment with speaker identity from diarization."""

    text: str
    start_ms: float
    end_ms: float
    speaker: str  # e.g. "SPEAKER_00", "SPEAKER_01"
    confidence: float | None = None


@dataclass
class DiarizedResult:
    """Full diarized transcription result — speaker-attributed segments."""

    segments: list[DiarizedSegment]
    language: str | None = None
    backend: str = "whisperx"
    num_speakers: int | None = None


class WhisperXBackend(TranscriptionBackend):
    """
    WhisperX transcription backend.

    - For stereo: call transcribe() per channel — fast, no diarization needed.
    - For mono: call transcribe_with_diarization() — runs pyannote to attribute speakers.

    Args:
        model_size: Whisper model size (tiny/base/small/medium/large-v2/large-v3)
        num_speakers: Hint for diarization. If None, auto-detected.
        device: "cuda" or "cpu". Auto-detected if None.
    """

    def __init__(
        self,
        model_size: str = "base",
        num_speakers: int | None = None,
        device: str | None = None,
    ):
        self.model_size = model_size
        self.num_speakers = num_speakers
        self.device = device
        self._model = None  # lazy load

    def _get_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _get_model(self):
        if self._model is None:
            try:
                import whisperx
            except ImportError as e:
                raise RuntimeError(
                    "whisperx not installed. Run: uv add whisperx"
                ) from e
            device = self._get_device()
            compute_type = "float16" if device == "cuda" else "int8"
            self._model = whisperx.load_model(self.model_size, device, compute_type=compute_type)
        return self._model

    def _prepare_samples(self, audio_samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Ensure float32 mono at 16kHz."""
        if audio_samples.ndim > 1:
            audio_samples = audio_samples.mean(axis=0)
        if sample_rate != 16000:
            import librosa
            audio_samples = librosa.resample(audio_samples, orig_sr=sample_rate, target_sr=16000)
        return audio_samples

    def transcribe(self, audio_samples: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """
        Transcribe a single audio channel (no diarization).
        Use this for stereo audio where speaker is already known from the channel.
        """
        import whisperx

        audio_samples = self._prepare_samples(audio_samples, sample_rate)
        model = self._get_model()
        result = model.transcribe(audio_samples, batch_size=16)

        segments = []
        for seg in result.get("segments", []):
            words = seg.get("words", [])
            probs = [w.get("score") for w in words if w.get("score") is not None]
            confidence = float(np.mean(probs)) if probs else None
            segments.append(TranscribedSegment(
                text=seg["text"].strip(),
                start_ms=seg["start"] * 1000,
                end_ms=seg["end"] * 1000,
                confidence=confidence,
                words=words,
            ))

        return TranscriptionResult(
            segments=segments,
            language=result.get("language"),
            backend="whisperx",
        )

    def transcribe_with_diarization(
        self, audio_samples: np.ndarray, sample_rate: int
    ) -> DiarizedResult:
        """
        Transcribe mono audio with speaker diarization.
        Requires HF_TOKEN env var for pyannote model access.
        """
        import whisperx

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN env var not set. WhisperX needs a HuggingFace token to download "
                "the pyannote diarization model. Set HF_TOKEN in your .env file.\n"
                "Accept the model license at: "
                "https://huggingface.co/pyannote/speaker-diarization-3.1"
            )

        device = self._get_device()
        audio_samples = self._prepare_samples(audio_samples, sample_rate)
        model = self._get_model()

        # Step 1: Transcribe
        result = model.transcribe(audio_samples, batch_size=16)
        language = result.get("language")

        # Step 2: Align word timestamps
        align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
        result = whisperx.align(result["segments"], align_model, metadata, audio_samples, device)

        # Step 3: Diarize
        from whisperx.diarize import DiarizationPipeline
        diarize_model = DiarizationPipeline(token=hf_token, device=device)
        diarize_kwargs = {}
        if self.num_speakers:
            diarize_kwargs["num_speakers"] = self.num_speakers
        diarize_segments = diarize_model(audio_samples, **diarize_kwargs)

        # Step 4: Assign speaker labels to words
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Step 5: Group consecutive words by speaker into segments
        diarized_segments: list[DiarizedSegment] = []
        current_speaker: str | None = None
        current_words: list[dict] = []
        current_start: float = 0.0

        for seg in result.get("segments", []):
            for word in seg.get("words", []):
                speaker = word.get("speaker") or seg.get("speaker") or "SPEAKER_00"
                if speaker != current_speaker:
                    if current_words and current_speaker is not None:
                        diarized_segments.append(self._flush_segment(
                            current_words, current_speaker, current_start
                        ))
                    current_speaker = speaker
                    current_words = [word]
                    current_start = word.get("start", 0) * 1000
                else:
                    current_words.append(word)

        if current_words and current_speaker is not None:
            diarized_segments.append(self._flush_segment(
                current_words, current_speaker, current_start
            ))

        speakers = {s.speaker for s in diarized_segments}
        return DiarizedResult(
            segments=diarized_segments,
            language=language,
            backend="whisperx",
            num_speakers=len(speakers),
        )

    def _flush_segment(
        self, words: list[dict], speaker: str, start_ms: float
    ) -> DiarizedSegment:
        text = " ".join(w["word"].strip() for w in words)
        end_ms = words[-1].get("end", 0) * 1000
        probs = [w["score"] for w in words if w.get("score") is not None]
        confidence = float(np.mean(probs)) if probs else None
        return DiarizedSegment(
            text=text, start_ms=start_ms, end_ms=end_ms,
            speaker=speaker, confidence=confidence,
        )


def transcribe_stereo(
    user_samples: np.ndarray,
    agent_samples: np.ndarray,
    sample_rate: int,
    backend: WhisperXBackend,
) -> tuple[TranscriptionResult, TranscriptionResult]:
    """
    Transcribe stereo audio by transcribing each channel independently.
    No diarization — speaker is known from channel position.
    Returns (user_result, agent_result).
    """
    user_result = backend.transcribe(user_samples, sample_rate)
    agent_result = backend.transcribe(agent_samples, sample_rate)
    return user_result, agent_result


def transcribe_with_diarization(
    samples: np.ndarray,
    sample_rate: int,
    backend: WhisperXBackend,
) -> tuple[DiarizedResult, dict[str, str]]:
    """
    Transcribe mono audio with WhisperX diarization.

    Returns (DiarizedResult, speaker_map) where speaker_map maps
    diarization speaker IDs (e.g. "SPEAKER_00") to "user" or "agent".
    Convention: first speaker heard = agent (typical for voice AI calls).
    """
    result = backend.transcribe_with_diarization(samples, sample_rate)

    seen: list[str] = []
    for seg in result.segments:
        if seg.speaker not in seen:
            seen.append(seg.speaker)
    speaker_map: dict[str, str] = {}
    for i, spk in enumerate(seen):
        speaker_map[spk] = "agent" if i == 0 else "user"

    return result, speaker_map


def merge_and_sort_turns(
    user_result: TranscriptionResult,
    agent_result: TranscriptionResult,
) -> list[tuple[str, TranscribedSegment]]:
    """
    Interleave user and agent segments sorted by start time.
    Returns list of (speaker_label, segment) tuples.
    """
    combined: list[tuple[str, TranscribedSegment]] = [
        ("user", seg) for seg in user_result.segments
    ] + [
        ("agent", seg) for seg in agent_result.segments
    ]
    combined.sort(key=lambda x: x[1].start_ms)
    return combined
