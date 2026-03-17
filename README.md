# Voice Evals

An open-source, tech-stack-independent framework for evaluating voice AI agents.

## What it does

Voice Evals takes a voice call recording (or a live webhook stream) and scores it across four dimensions:

| Metric | What it measures |
|---|---|
| **Conversation Quality** | Does the agent use good voice-first communication? Short sentences, no list-reading, natural phrasing |
| **Multi-turn Coherence** | Does the agent maintain context across turns? No repeated questions, correct references |
| **Intent Accuracy** | Did the agent correctly interpret what the user was asking each turn? |
| **Task Completion** | Did the scripted task get completed, end-to-end? |

## Quick start

```bash
# Install uv (if you haven't already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up
git clone https://github.com/your-org/voice-evals
cd voice-evals

# Create venv + install all dependencies (including dev)
uv sync

# Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Evaluate a recording
uv run voice-evals evaluate path/to/recording.wav

# With a scenario config for richer scoring
uv run voice-evals evaluate recording.wav --scenario scenarios/restaurant-booking.yaml

# Run tests
uv run pytest

# Start the API server
cd backend && uv run uvicorn api.main:app --reload --port 8000

# Start the frontend
cd frontend && npm install && npm run dev
```

## Project structure

```
voice-evals/
├── backend/
│   ├── voice_evals/
│   │   ├── trace.py              # VoiceTrace schema (canonical data model)
│   │   ├── ingestion/
│   │   │   ├── audio.py          # Load WAV/MP3/OGG, split channels
│   │   │   └── transcribe.py     # Whisper transcription backend
│   │   ├── metrics/
│   │   │   ├── conversation_quality.py
│   │   │   ├── coherence.py
│   │   │   ├── intent.py
│   │   │   └── task_completion.py
│   │   ├── evaluator.py          # Orchestrates full evaluation run
│   │   └── cli.py                # CLI (voice-evals command)
│   └── api/
│       ├── main.py               # FastAPI app
│       └── routes/
│           ├── evaluate.py       # POST /api/v1/evaluate
│           └── reports.py        # GET /api/v1/reports/:id
├── frontend/                     # React + TypeScript UI
├── scenarios/                    # Example YAML scenario configs
└── tests/
```

## Two evaluation modes

### Mode 1: Upload & Evaluate
Upload any audio recording. No platform integration needed.

```bash
voice-evals evaluate recording.wav --model base
```

**Stereo audio** (left = user, right = agent) gives cleaner per-speaker analysis. Mono audio uses VAD-based turn segmentation as a fallback.

### Mode 2: Live / Webhook (coming soon)
Platform connectors for Vapi, Retell, Twilio. Send webhook events to `POST /api/v1/webhooks/:platform` to build a VoiceTrace in real-time.

## Scenario configs

YAML scenario configs unlock richer scoring:

```yaml
scenario_id: restaurant-booking-v1
expected_task: Book a dinner reservation for 2 at 7:30pm on Friday
completion_criteria: Agent confirms party size, date/time, and contact info
expected_intents:
  - request_reservation
  - provide_party_size
  - provide_date_time
  - provide_contact_info
```

See `scenarios/` for examples.

## Metric tiers

### Tier 1 (implemented): LLM-judge metrics
Work from any recording or transcript. Use Claude as the judge.

### Tier 2 (planned): Timing metrics
- **TTFW** (Time to First Word): user speech end → agent first audio byte
- **Interruption recovery rate**: % of barge-ins handled gracefully
- **VAD false positive rate**: noise-triggered activations

### Tier 3 (planned): Audio signal quality
- **WER** (Word Error Rate): STT accuracy against ground truth
- **MOS** (Mean Opinion Score): does the agent sound like a robot?
- **SNR**: signal-to-noise ratio

## License

MIT
