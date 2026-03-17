"""
Audio ingestion: load audio files, detect format/metadata, split stereo channels.

Supports WAV, MP3, OGG, M4A, FLAC via pydub (ffmpeg backend).
Returns numpy arrays for downstream processing.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LoadedAudio:
    """Audio loaded from disk, ready for analysis."""

    samples: np.ndarray       # shape: (n_samples,) mono or (2, n_samples) stereo
    sample_rate: int
    n_channels: int
    duration_ms: float
    format: str
    file_path: str

    @property
    def is_stereo(self) -> bool:
        return self.n_channels == 2

    @property
    def user_channel(self) -> np.ndarray | None:
        """Left channel (index 0) — assumed to be the user."""
        if self.is_stereo:
            return self.samples[0]
        return None

    @property
    def agent_channel(self) -> np.ndarray | None:
        """Right channel (index 1) — assumed to be the agent."""
        if self.is_stereo:
            return self.samples[1]
        return None

    @property
    def mono_mix(self) -> np.ndarray:
        """Downmix to mono for processing that doesn't need speaker separation."""
        if self.is_stereo:
            return (self.samples[0] + self.samples[1]) / 2.0
        return self.samples


def load_audio(file_path: str | Path, target_sr: int = 16000) -> LoadedAudio:
    """
    Load an audio file and return a LoadedAudio instance.

    Args:
        file_path: Path to the audio file (WAV, MP3, OGG, M4A, FLAC).
        target_sr: Resample to this sample rate (default 16kHz for Whisper/VAD).

    Returns:
        LoadedAudio with normalized float32 samples in [-1, 1].
    """
    try:
        import librosa
        import soundfile as sf
    except ImportError as e:
        raise RuntimeError(
            "Audio dependencies not installed. Run: pip install librosa soundfile pydub"
        ) from e

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    fmt = file_path.suffix.lstrip(".").lower()
    if fmt not in {"wav", "mp3", "ogg", "m4a", "flac", "aac", "webm"}:
        raise ValueError(f"Unsupported audio format: {fmt}")

    # Convert non-WAV to WAV in a temp file for librosa
    if fmt != "wav":
        wav_path = _convert_to_wav(file_path)
        load_path = wav_path
    else:
        load_path = file_path
        wav_path = None

    try:
        # Load with librosa — handles resampling + format normalization
        # mono=False preserves stereo channels
        samples, sr = librosa.load(str(load_path), sr=target_sr, mono=False, dtype=np.float32)
    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)

    # librosa returns (n_samples,) for mono, (n_channels, n_samples) for stereo
    if samples.ndim == 1:
        n_channels = 1
    else:
        n_channels = samples.shape[0]

    duration_ms = (samples.shape[-1] / sr) * 1000.0

    return LoadedAudio(
        samples=samples,
        sample_rate=sr,
        n_channels=n_channels,
        duration_ms=duration_ms,
        format=fmt,
        file_path=str(file_path),
    )


def split_channels(audio: LoadedAudio) -> tuple[np.ndarray, np.ndarray]:
    """
    Split stereo audio into (user_samples, agent_samples).

    Convention: left channel (0) = user, right channel (1) = agent.
    For mono audio, both returned arrays are the same mono mix.
    """
    if audio.is_stereo:
        return audio.samples[0], audio.samples[1]
    return audio.samples, audio.samples


def extract_segment(audio: LoadedAudio, start_ms: float, end_ms: float) -> np.ndarray:
    """
    Extract a time segment from audio samples.

    Args:
        audio: Loaded audio.
        start_ms: Start time in milliseconds.
        end_ms: End time in milliseconds.

    Returns:
        Numpy array of samples for the segment.
    """
    start_sample = int(start_ms * audio.sample_rate / 1000)
    end_sample = int(end_ms * audio.sample_rate / 1000)

    if audio.is_stereo:
        return audio.samples[:, start_sample:end_sample]
    return audio.samples[start_sample:end_sample]


def save_channel_wav(samples: np.ndarray, sample_rate: int, output_path: str) -> None:
    """Save a single-channel numpy array as a WAV file."""
    import soundfile as sf

    sf.write(output_path, samples, sample_rate, subtype="PCM_16")


def _convert_to_wav(file_path: Path) -> str:
    """Convert non-WAV audio to a temporary WAV file via pydub."""
    try:
        from pydub import AudioSegment as PydubSegment
    except ImportError as e:
        raise RuntimeError("pydub not installed. Run: pip install pydub") from e

    audio = PydubSegment.from_file(str(file_path))
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(tmp.name, format="wav")
    tmp.close()
    return tmp.name
