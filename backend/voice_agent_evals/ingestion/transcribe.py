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

import json
import logging
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
    is_overlap: bool = False  # True if pyannote detected overlapping speech in this window


@dataclass
class DiarizedResult:
    """Full diarized transcription result — speaker-attributed segments."""

    segments: list[DiarizedSegment]
    language: str | None = None
    backend: str = "whisperx"
    num_speakers: int | None = None
    overlap_regions_ms: list[tuple[float, float]] = field(default_factory=list)  # (start_ms, end_ms)


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

        # Step 6: Overlap detection via pyannote
        overlap_regions_ms: list[tuple[float, float]] = []
        try:
            from pyannote.audio import Pipeline as PyannotePipeline
            overlap_pipeline = PyannotePipeline.from_pretrained(
                "pyannote/overlapped-speech-detection",
                use_auth_token=hf_token,
            )
            import torch
            audio_tensor = torch.tensor(audio_samples).unsqueeze(0)  # (1, samples)
            overlap_result = overlap_pipeline({
                "waveform": audio_tensor,
                "sample_rate": 16000,
            })
            for segment, _, _ in overlap_result.itertracks(yield_label=True):
                overlap_regions_ms.append((segment.start * 1000, segment.end * 1000))
        except Exception:
            # Overlap detection is best-effort — don't fail the whole transcription
            pass

        # Flag segments that fall within an overlap region (>200ms overlap threshold)
        for seg in diarized_segments:
            for ov_start, ov_end in overlap_regions_ms:
                overlap_ms = min(seg.end_ms, ov_end) - max(seg.start_ms, ov_start)
                if overlap_ms >= 200:
                    seg.is_overlap = True
                    break

        speakers = {s.speaker for s in diarized_segments}
        return DiarizedResult(
            segments=diarized_segments,
            language=language,
            backend="whisperx",
            num_speakers=len(speakers),
            overlap_regions_ms=overlap_regions_ms,
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


def classify_speakers_with_llm(
    segments: list[DiarizedSegment],
    scenario_task: str | None = None,
) -> dict[str, str]:
    """
    Use Claude to classify which diarization speaker ID is the agent vs user.

    Sends only the first ~10 segments. Fast and cheap but assumes diarization
    speaker IDs are consistent (one ID = one real speaker).

    Returns {"agent_speaker": "SPEAKER_XX", "user_speaker": "SPEAKER_YY"} on success,
    or an empty dict on failure/low confidence (caller should fallback).
    """
    try:
        import anthropic
    except ImportError:
        logging.warning("anthropic package not installed — skipping LLM speaker classification")
        return {}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logging.warning("ANTHROPIC_API_KEY not set — skipping LLM speaker classification")
        return {}

    excerpt_lines = [f"{seg.speaker}: {seg.text}" for seg in segments[:10]]
    excerpt = "\n".join(excerpt_lines)
    scenario_hint = (
        f"\nThe expected task for this call is: {scenario_task}" if scenario_task else ""
    )

    prompt = (
        f"You are analyzing a voice call transcript excerpt to determine which speaker "
        f"is the AI agent and which is the human user.{scenario_hint}\n\n"
        f"Transcript excerpt (first turns of the call):\n{excerpt}\n\n"
        f"Agent characteristics: formal greetings (\"How can I help you today?\"), "
        f"structured responses, scripted language, offers assistance.\n"
        f"User characteristics: makes requests, asks questions, provides personal "
        f"information, more conversational tone.\n\n"
        f"Respond with ONLY valid JSON in this exact format:\n"
        f'{{\"agent_speaker\": \"SPEAKER_XX\", \"user_speaker\": \"SPEAKER_YY\", '
        f'\"confidence\": \"high\", \"reasoning\": \"brief explanation\"}}\n\n'
        f"Use \"low\" for confidence if you cannot determine roles clearly."
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=os.environ.get("VOICE_EVALS_MODEL", "claude-sonnet-4-6"),
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = message.content[0].text.strip()
        if response_text.startswith("```"):
            parts = response_text.split("```")
            response_text = parts[1] if len(parts) > 1 else response_text
            if response_text.startswith("json"):
                response_text = response_text[4:]
        result = json.loads(response_text.strip())

        if result.get("confidence") == "low":
            logging.warning(
                "LLM speaker classification returned low confidence: %s — "
                "falling back to first-speaker=agent",
                result.get("reasoning", ""),
            )
            return {}

        agent_spk = result.get("agent_speaker")
        user_spk = result.get("user_speaker")
        if not agent_spk or not user_spk:
            logging.warning("LLM speaker classification returned incomplete result — falling back")
            return {}

        logging.info(
            "LLM speaker classification: agent=%s, user=%s — %s",
            agent_spk, user_spk, result.get("reasoning", ""),
        )
        return {"agent_speaker": agent_spk, "user_speaker": user_spk}
    except Exception as e:
        logging.warning(
            "LLM speaker classification failed (%s) — falling back to first-speaker=agent", e
        )
        return {}


def assign_roles_per_turn_with_llm(
    segments: list[DiarizedSegment],
    scenario_task: str | None = None,
) -> list[str] | None:
    """
    Use Claude to assign "agent" or "user" to each diarized segment individually.

    Sends the full transcript to Claude and asks it to label each turn by content,
    ignoring diarization speaker IDs. This handles diarization errors where the same
    real speaker appears under multiple IDs, or multiple real speakers share one ID.

    Returns a list of role strings (["agent", "user", ...]) aligned to segments,
    or None on failure (caller should fallback to first-speaker heuristic).
    """
    try:
        import anthropic
    except ImportError:
        logging.warning("anthropic package not installed — skipping LLM speaker classification")
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logging.warning("ANTHROPIC_API_KEY not set — skipping LLM speaker classification")
        return None

    scenario_hint = (
        f"\nThe expected task for this call is: {scenario_task}" if scenario_task else ""
    )

    turns_text = "\n".join(
        f"{i + 1} [{seg.speaker}]: {seg.text}" for i, seg in enumerate(segments)
    )

    prompt = (
        f"You are analyzing a diarized voice call transcript. Speaker IDs (SPEAKER_XX) "
        f"are unreliable — the same real speaker may appear under different IDs, and one "
        f"ID may contain turns from different real speakers. Ignore the IDs entirely and "
        f"assign roles based on what each turn actually says.{scenario_hint}\n\n"
        f"Agent characteristics: hold/connect messages, formal greetings, scripted "
        f"language, offers assistance, provides information.\n"
        f"User characteristics: responds to questions, makes requests, provides personal "
        f"details, more conversational.\n\n"
        f"Transcript:\n{turns_text}\n\n"
        f"Respond with ONLY a JSON array with one entry per turn in order:\n"
        f'[{{"turn": 1, "role": "agent"}}, {{"turn": 2, "role": "user"}}, ...]\n\n'
        f"Every turn must have a role. Use exactly the strings \"agent\" or \"user\"."
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=os.environ.get("VOICE_EVALS_MODEL", "claude-sonnet-4-6"),
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = message.content[0].text.strip()
        if response_text.startswith("```"):
            parts = response_text.split("```")
            response_text = parts[1] if len(parts) > 1 else response_text
            if response_text.startswith("json"):
                response_text = response_text[4:]
        result = json.loads(response_text.strip())

        if not isinstance(result, list) or len(result) != len(segments):
            logging.warning(
                "LLM per-turn classification returned %d entries for %d segments — falling back",
                len(result) if isinstance(result, list) else -1,
                len(segments),
            )
            return None

        roles = []
        for entry in result:
            role = entry.get("role", "").lower()
            if role not in ("agent", "user"):
                logging.warning("LLM returned unexpected role '%s' — falling back", role)
                return None
            roles.append(role)

        logging.info("LLM per-turn classification complete: %s", roles)
        return roles
    except Exception as e:
        logging.warning(
            "LLM per-turn classification failed (%s) — falling back to first-speaker=agent", e
        )
        return None


def _build_speaker_map(seen: list[str], first_speaker: str) -> dict[str, str]:
    """Build a speaker_map from ordered speaker IDs using the first-speaker convention."""
    speaker_map: dict[str, str] = {}
    if first_speaker == "user":
        for i, spk in enumerate(seen):
            speaker_map[spk] = "user" if i == 0 else "agent"
    else:  # "agent" (default/fallback)
        for i, spk in enumerate(seen):
            speaker_map[spk] = "agent" if i == 0 else "user"
    return speaker_map


def transcribe_with_diarization(
    samples: np.ndarray,
    sample_rate: int,
    backend: WhisperXBackend,
    first_speaker: str = "auto",
    scenario_task: str | None = None,
) -> tuple[DiarizedResult, dict[str, str]]:
    """
    Transcribe mono audio with WhisperX diarization.

    Returns (DiarizedResult, speaker_map) where speaker_map maps
    diarization speaker IDs (e.g. "SPEAKER_00") to "user" or "agent".

    Args:
        first_speaker: "auto" (LLM speaker-ID classification), "llm" (per-turn LLM
                       assignment), "agent" (first heard = agent), or "user" (first
                       heard = user).
        scenario_task: Optional expected task hint passed to LLM classification.
    """
    result = backend.transcribe_with_diarization(samples, sample_rate)

    seen: list[str] = []
    for seg in result.segments:
        if seg.speaker not in seen:
            seen.append(seg.speaker)

    if first_speaker == "llm":
        per_turn_roles = assign_roles_per_turn_with_llm(result.segments, scenario_task)
        if per_turn_roles is not None:
            for seg, role in zip(result.segments, per_turn_roles):
                seg.speaker = role
            speaker_map = {role: role for role in ("agent", "user")}
        else:
            speaker_map = _build_speaker_map(seen, "agent")
    elif first_speaker == "auto":
        llm_result = classify_speakers_with_llm(result.segments, scenario_task)
        if llm_result:
            agent_spk = llm_result["agent_speaker"]
            user_spk = llm_result["user_speaker"]
            speaker_map = {}
            for spk in seen:
                if spk == agent_spk:
                    speaker_map[spk] = "agent"
                elif spk == user_spk:
                    speaker_map[spk] = "user"
                else:
                    speaker_map[spk] = "user"
        else:
            speaker_map = _build_speaker_map(seen, "agent")
    else:
        speaker_map = _build_speaker_map(seen, first_speaker)

    return result, speaker_map


class AssemblyAIBackend:
    """
    AssemblyAI transcription + speaker diarization via cloud API.

    Uploads the audio file to AssemblyAI, runs transcription and speaker
    diarization server-side, and returns speaker-attributed utterances.

    Requires:
        - pip install assemblyai
        - ASSEMBLYAI_API_KEY env var

    Args:
        num_speakers: Optional hint for number of speakers. If None, auto-detected.
    """

    def __init__(self, num_speakers: int | None = None):
        self.num_speakers = num_speakers

    def transcribe_with_diarization(
        self, audio_path: str, sample_rate: int | None = None
    ) -> DiarizedResult:
        """
        Transcribe an audio file with speaker diarization via AssemblyAI API.
        Takes a file path (not raw samples) since AssemblyAI handles format conversion.
        """
        try:
            import assemblyai as aai
        except ImportError as e:
            raise RuntimeError(
                "assemblyai not installed. Run: uv add assemblyai"
            ) from e

        api_key = os.environ.get("ASSEMBLYAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ASSEMBLYAI_API_KEY env var not set. Add it to your .env file."
            )

        aai.settings.api_key = api_key

        config = aai.TranscriptionConfig(
            speaker_labels=True,
            speakers_expected=self.num_speakers,
            speech_models=["universal-3-pro", "universal-2"],
            language_detection=True,
        )

        transcript = aai.Transcriber(config=config).transcribe(audio_path)

        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")

        diarized_segments: list[DiarizedSegment] = []
        for utt in transcript.utterances or []:
            diarized_segments.append(DiarizedSegment(
                text=utt.text,
                start_ms=float(utt.start),
                end_ms=float(utt.end),
                speaker=f"SPEAKER_{utt.speaker}",
                confidence=utt.confidence,
            ))

        speakers = {s.speaker for s in diarized_segments}
        return DiarizedResult(
            segments=diarized_segments,
            language=transcript.language_code,
            backend="assemblyai",
            num_speakers=len(speakers),
        )


def transcribe_with_assemblyai(
    audio_path: str,
    backend: AssemblyAIBackend,
    first_speaker: str = "auto",
    scenario_task: str | None = None,
) -> tuple[DiarizedResult, dict[str, str]]:
    """
    Transcribe an audio file with AssemblyAI diarization.

    Returns (DiarizedResult, speaker_map) where speaker_map maps
    AssemblyAI speaker IDs (e.g. "SPEAKER_A") to "user" or "agent".

    Args:
        first_speaker: "auto" (LLM speaker-ID classification), "llm" (per-turn LLM
                       assignment), "agent" (first heard = agent), or "user" (first
                       heard = user).
        scenario_task: Optional expected task hint passed to LLM classification.
    """
    result = backend.transcribe_with_diarization(audio_path)

    seen: list[str] = []
    for seg in result.segments:
        if seg.speaker not in seen:
            seen.append(seg.speaker)

    if first_speaker == "llm":
        per_turn_roles = assign_roles_per_turn_with_llm(result.segments, scenario_task)
        if per_turn_roles is not None:
            for seg, role in zip(result.segments, per_turn_roles):
                seg.speaker = role
            speaker_map = {role: role for role in ("agent", "user")}
        else:
            speaker_map = _build_speaker_map(seen, "agent")
    elif first_speaker == "auto":
        llm_result = classify_speakers_with_llm(result.segments, scenario_task)
        if llm_result:
            agent_spk = llm_result["agent_speaker"]
            user_spk = llm_result["user_speaker"]
            speaker_map = {}
            for spk in seen:
                if spk == agent_spk:
                    speaker_map[spk] = "agent"
                elif spk == user_spk:
                    speaker_map[spk] = "user"
                else:
                    speaker_map[spk] = "user"
        else:
            speaker_map = _build_speaker_map(seen, "agent")
    else:
        speaker_map = _build_speaker_map(seen, first_speaker)

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
