# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# General rules of behaviour
1. Before writing any code, describe your approach and wait for approval. Always ask clarifying questions before writing any code if requirements ar      e     ambiguous.
2. If a task requires changes to more than 3 files, stop and break it into smaller tasks first.
3. After writing code, list what could break and suggest tests to cover it.
4. When there’s a bug, start by writing a test that reproduces it, then fix it until the test passes.
5. Every time I correct you, add a new rule to the CLAUDE .md file so it never happens again.
6. Everytime you start a new feature, create a new branch so we can track

## Project Overview

**Voice Evals** is an open-source framework for evaluating voice AI agents. It takes a voice call recording and scores it across four dimensions using Claude as an LLM judge: Conversation Quality, Multi-turn Coherence, Intent Accuracy, and Task Completion.

## Commands

### Backend (Python)

```bash
# Install dependencies
uv sync

# Run CLI evaluation
uv run voice-evals evaluate path/to/recording.wav
uv run voice-evals evaluate recording.wav --scenario scenarios/restaurant-booking.yaml --verbose

# Start API server
cd backend && uv run uvicorn api.main:app --reload --port 8000

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_evaluator.py

# Lint
uv run ruff check .

# Type check
uv run mypy backend/voice_evals
```

### Frontend (React/TypeScript)

```bash
cd frontend
npm install
npm run dev    # dev server on port 5173, proxies /api to localhost:8000
npm run build
```

### Environment

```bash
cp .env.example .env
# Required: ANTHROPIC_API_KEY
# Optional: VOICE_EVALS_MODEL (default: claude-sonnet-4-6), WHISPER_MODEL (default: base)
```

## Architecture

### Core Data Flow

```
Audio File → load_audio() → [split_channels()] → WhisperBackend.transcribe()
    → build_trace() → VoiceTrace → Evaluator.run() → EvaluationReport
```

**VoiceTrace** (`backend/voice_evals/trace.py`) is the canonical platform-agnostic schema — the contract between ingestion and evaluation. All metrics operate on this model.

### Backend Structure

- **`voice_evals/ingestion/audio.py`** — Loads WAV/MP3/OGG/M4A/FLAC, normalizes to 16kHz float32. For stereo: left channel = user, right channel = agent.
- **`voice_evals/ingestion/transcribe.py`** — Pluggable `TranscriptionBackend` ABC; `WhisperBackend` is the default implementation. `transcribe_stereo()` processes each channel separately for clean speaker attribution.
- **`voice_evals/metrics/base.py`** — `BaseMetric` ABC with `evaluate(trace) -> MetricResult`. Contains `call_llm_judge()` and `parse_score_response()` utilities shared by all metrics.
- **`voice_evals/metrics/`** — Four Tier 1 metrics: `conversation_quality`, `coherence`, `intent`, `task_completion`. Each calls Claude as a judge and returns a score (0.0–1.0) with reasoning.
- **`voice_evals/evaluator.py`** — Orchestrates all metrics; one metric failure is isolated and does not abort others.
- **`voice_evals/cli.py`** — Typer CLI; entry point is `voice-evals evaluate`.
- **`api/`** — FastAPI app. `POST /api/v1/evaluate` accepts audio + optional scenario YAML (max 100MB). Reports stored in-memory (swap for DB in production).

### Metric Tiers

- **Tier 1 (implemented)**: LLM-judge metrics (the four above)
- **Tier 2 (planned)**: Timing metrics (latency, TTFW)
- **Tier 3 (planned)**: Audio signal quality

### Scenarios

Optional YAML files (`scenarios/`) define `expected_task`, `completion_criteria`, and `expected_intents`. Providing a scenario enables richer evaluation for Intent Accuracy and Task Completion metrics.

### Frontend

React 18 + TypeScript + Vite + Tailwind. Dev server proxies `/api` to backend. Types in `src/types.ts` mirror backend Pydantic models. Views: `upload` → `report` → `history`.

## Key Conventions

- All Python package management via `uv` — never use `pip` or `venv` directly.
- Ruff line length is 100; import style enforced (E, F, I, UP rules).
- New transcription backends should implement `TranscriptionBackend` ABC.
- New metrics should extend `BaseMetric` and use `call_llm_judge()` / `parse_score_response()` from `base.py`.
