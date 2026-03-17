"""
POST /evaluate — accepts audio file upload + optional scenario config,
runs the full evaluation pipeline, and returns or stores the report.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from typing import Annotated

import yaml
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from voice_evals.evaluator import Evaluator
from voice_evals.ingestion.audio import load_audio, split_channels
from voice_evals.ingestion.transcribe import (
    WhisperXBackend,
    merge_and_sort_turns,
    transcribe_stereo,
    transcribe_with_diarization,
)
from voice_evals.trace import (
    AudioInfo,
    PlatformInfo,
    ScenarioConfig,
    Speaker,
    TimingInfo,
    Turn,
    VoiceTrace,
)

router = APIRouter()

# In-memory store for demo (swap for a real DB in production)
_reports: dict[str, dict] = {}

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".m4a", ".flac"}
MAX_UPLOAD_SIZE_MB = 100


@router.post("/evaluate")
async def evaluate_audio(
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, M4A, FLAC)"),
    scenario_yaml: str | None = Form(None, description="Optional YAML scenario config"),
    whisper_model: str = Form("base", description="Whisper model size: tiny/base/small/medium"),
    embed_trace: bool = Form(False, description="Embed full trace in response"),
):
    """
    Upload an audio recording and run full voice evaluation.

    Returns an EvaluationReport with per-metric scores and a summary.
    """
    # Validate file extension
    ext = os.path.splitext(audio.filename or "")[1].lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{ext}'. Supported: {ALLOWED_AUDIO_EXTENSIONS}",
        )

    # Read uploaded bytes
    content = await audio.read()
    if len(content) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE_MB}MB.",
        )

    # Parse optional scenario config
    scenario: ScenarioConfig | None = None
    if scenario_yaml:
        try:
            data = yaml.safe_load(scenario_yaml)
            scenario = ScenarioConfig(**data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid scenario YAML: {e}")

    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        trace = await _build_trace(tmp_path, scenario, whisper_model)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Audio processing failed: {e}")
    finally:
        os.unlink(tmp_path)

    # Run evaluation
    evaluator = Evaluator(embed_trace=embed_trace)
    report = evaluator.run(trace)

    # Store report
    _reports[report.report_id] = report.model_dump()

    return JSONResponse(
        status_code=200,
        content=report.model_dump(mode="json"),
    )


async def _build_trace(
    audio_path: str,
    scenario: ScenarioConfig | None,
    whisper_model: str,
) -> VoiceTrace:
    """Load audio, transcribe, and build a VoiceTrace."""
    audio = load_audio(audio_path)

    trace = VoiceTrace(
        audio_info=AudioInfo(
            original_file=os.path.basename(audio_path),
            duration_ms=audio.duration_ms,
            sample_rate=audio.sample_rate,
            channels=audio.n_channels,
            format=audio.format,
            user_channel=0 if audio.is_stereo else None,
            agent_channel=1 if audio.is_stereo else None,
        ),
        platform_info=PlatformInfo(platform="upload"),
        scenario=scenario,
    )

    backend = WhisperXBackend(model_size=whisper_model)

    if audio.is_stereo:
        user_samples, agent_samples = split_channels(audio)
        user_result, agent_result = transcribe_stereo(
            user_samples, agent_samples, audio.sample_rate, backend
        )
        interleaved = merge_and_sort_turns(user_result, agent_result)
        for speaker_label, seg in interleaved:
            trace.add_turn(Turn(
                speaker=Speaker.USER if speaker_label == "user" else Speaker.AGENT,
                transcript=seg.text,
                transcript_confidence=seg.confidence,
                timing=TimingInfo(
                    speech_start_ms=seg.start_ms,
                    speech_end_ms=seg.end_ms,
                    source="vad",
                ),
            ))
    else:
        diarized, speaker_map = transcribe_with_diarization(
            audio.mono_mix, audio.sample_rate, backend
        )
        for seg in diarized.segments:
            spk = speaker_map.get(seg.speaker, "user")
            trace.add_turn(Turn(
                speaker=Speaker.AGENT if spk == "agent" else Speaker.USER,
                transcript=seg.text,
                transcript_confidence=seg.confidence,
                timing=TimingInfo(
                    speech_start_ms=seg.start_ms,
                    speech_end_ms=seg.end_ms,
                    source="vad",
                ),
            ))

    return trace
