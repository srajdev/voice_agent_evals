"""
Transcription backend: convert audio → per-turn transcript.

Default backend: OpenAI Whisper (local, free).
Pluggable: implement TranscriptionBackend to add Deepgram, AssemblyAI, etc.

For stereo audio, transcribes each channel separately to get clean per-speaker turns.
For mono audio, uses Whisper's word-level timestamps to segment turns (best-effort).

WhisperXBackend: uses whisperx (Whisper + pyannote diarization) for accurate
speaker attribution on mono audio. Requires HF_TOKEN env var.
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


class WhisperXBackend:
    """
    WhisperX transcription + speaker diarization.

    Combines Whisper (transcription) with pyannote (diarization) to produce
    speaker-attributed segments from mono audio. Much more accurate than the
    alternating-index heuristic used by WhisperBackend on mono audio.

    Requires:
        - pip install whisperx
        - HF_TOKEN env var set to a HuggingFace token with pyannote model access
          (accept license at https://huggingface.co/pyannote/speaker-diarization-3.1)

    Args:
        model_size: Whisper model size (tiny/base/small/medium/large-v2/large-v3)
        num_speakers: Optional hint for number of speakers. If None, auto-detected.
        device: "cuda" or "cpu". Defaults to cuda if available, else cpu.
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
                    "whisperx not installed. Run: pip install whisperx"
                ) from e
            device = self._get_device()
            compute_type = "float16" if device == "cuda" else "int8"
            self._model = whisperx.load_model(self.model_size, device, compute_type=compute_type)
        return self._model

    def transcribe_with_diarization(
        self, audio_samples: np.ndarray, sample_rate: int
    ) -> DiarizedResult:
        """
        Transcribe mono audio and return speaker-attributed segments.

        This is separate from the TranscriptionBackend.transcribe() interface
        because diarization produces speaker labels rather than per-channel results.
        """
        try:
            import whisperx
        except ImportError as e:
            raise RuntimeError("whisperx not installed. Run: pip install whisperx") from e

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN env var not set. WhisperX needs a HuggingFace token to download "
                "the pyannote diarization model. Set HF_TOKEN in your .env file.\n"
                "You also need to accept the model license at: "
                "https://huggingface.co/pyannote/speaker-diarization-3.1"
            )

        device = self._get_device()
        model = self._get_model()

        # Ensure float32 mono at 16kHz
        if audio_samples.ndim > 1:
            audio_samples = audio_samples.mean(axis=0)
        if sample_rate != 16000:
            import librosa
            audio_samples = librosa.resample(audio_samples, orig_sr=sample_rate, target_sr=16000)

        # Step 1: Transcribe with Whisper via whisperx
        result = model.transcribe(audio_samples, batch_size=16)
        language = result.get("language")

        # Step 2: Align word timestamps
        align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
        result = whisperx.align(result["segments"], align_model, metadata, audio_samples, device)

        # Step 3: Diarize
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        diarize_kwargs = {}
        if self.num_speakers:
            diarize_kwargs["num_speakers"] = self.num_speakers
        diarize_segments = diarize_model(audio_samples, **diarize_kwargs)

        # Step 4: Assign speaker labels to words
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Step 5: Build DiarizedSegments — group consecutive words by speaker
        diarized_segments: list[DiarizedSegment] = []
        current_speaker: str | None = None
        current_words: list[dict] = []
        current_start: float = 0.0

        for seg in result.get("segments", []):
            for word in seg.get("words", []):
                speaker = word.get("speaker") or seg.get("speaker") or "SPEAKER_00"
                if speaker != current_speaker:
                    # Flush current group
                    if current_words and current_speaker is not None:
                        text = " ".join(w["word"].strip() for w in current_words)
                        end_ms = current_words[-1].get("end", 0) * 1000
                        probs = [w["score"] for w in current_words if w.get("score") is not None]
                        confidence = float(np.mean(probs)) if probs else None
                        diarized_segments.append(DiarizedSegment(
                            text=text,
                            start_ms=current_start,
                            end_ms=end_ms,
                            speaker=current_speaker,
                            confidence=confidence,
                        ))
                    current_speaker = speaker
                    current_words = [word]
                    current_start = word.get("start", 0) * 1000
                else:
                    current_words.append(word)

        # Flush last group
        if current_words and current_speaker is not None:
            text = " ".join(w["word"].strip() for w in current_words)
            end_ms = current_words[-1].get("end", 0) * 1000
            probs = [w["score"] for w in current_words if w.get("score") is not None]
            confidence = float(np.mean(probs)) if probs else None
            diarized_segments.append(DiarizedSegment(
                text=text,
                start_ms=current_start,
                end_ms=end_ms,
                speaker=current_speaker,
                confidence=confidence,
            ))

        speakers = {s.speaker for s in diarized_segments}
        return DiarizedResult(
            segments=diarized_segments,
            language=language,
            backend="whisperx",
            num_speakers=len(speakers),
        )


def transcribe_with_whisperx(
    samples: np.ndarray,
    sample_rate: int,
    backend: WhisperXBackend,
) -> tuple[DiarizedResult, dict[str, str]]:
    """
    Transcribe mono audio with WhisperX diarization.

    Returns (DiarizedResult, speaker_map) where speaker_map maps
    diarization speaker IDs (e.g. "SPEAKER_00") to "user" or "agent".
    The first speaker heard is assumed to be the agent (typical for voice AI calls).
    """
    result = backend.transcribe_with_diarization(samples, sample_rate)

    # Map speakers to user/agent by order of first appearance
    # Convention: first speaker = agent (they typically speak first on AI calls)
    seen: list[str] = []
    for seg in result.segments:
        if seg.speaker not in seen:
            seen.append(seg.speaker)
    speaker_map: dict[str, str] = {}
    for i, spk in enumerate(seen):
        speaker_map[spk] = "agent" if i == 0 else "user"

    return result, speaker_map


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
