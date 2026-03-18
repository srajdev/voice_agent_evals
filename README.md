# Voice Agent Evals

An open-source, tech-stack-independent framework for evaluating voice AI agents.

## What it does

Voice Agent Evals takes a voice call recording (or a live webhook stream) and scores it across three categories of metrics using Claude as an LLM judge.

### Outcome Metrics — *Did the conversation go well?*

LLM-judge metrics that work from any recording or transcript.

| Metric | What it measures |
|---|---|
| **Conversation Quality** | Does the agent use good voice-first communication? Scores for short natural sentences, no list-reading, appropriate length, and avoiding robotic phrasing. |
| **Multi-turn Coherence** | Does the agent maintain context across turns? Penalises asking for info the user already gave, broken references, and contradictions between turns. |
| **Intent Accuracy** | Did the agent correctly interpret the user's intent each turn? Uses expected intents from a scenario config if provided, otherwise infers them. |
| **Task Completion** | Did the agent complete the scripted task end-to-end? Scores 0.0–1.0 with partial credit; requires a scenario config for best results. |

### Technical Metrics — *How did the agent perform under the hood?*

Timing and signal-integrity metrics derived from audio timestamps and VAD data.

| Metric | What it measures |
|---|---|
| **Response Latency** | Time between the user finishing speaking and the agent starting to respond. Thresholds: excellent <800 ms, good <1500 ms, fair <2500 ms, poor ≥2500 ms. |
| **VAD Quality** | Rate of VAD false positives — background noise or silence mistakenly triggering the agent. Heuristically flags short, low-confidence agent turns, then confirms with an LLM judge. |
| **Interruption Recovery** | How gracefully the agent handles user barge-ins. Detects overlaps via timing data or platform metadata, then scores each event as fully recovered, partial, or missed. |

### Quality Metrics — *Does the agent communicate like a human?*

LLM-judged style metrics that assess how well the agent adapts its language to the user.

| Metric | What it measures |
|---|---|
| **Verbosity Match** | Are agent response lengths proportional to what the user said? Blends a ratio-based algorithmic score with an LLM contextual review. Ideal agent-to-user word ratio: 0.5–2.5×. |
| **Empathy** | Does the agent acknowledge emotional signals? Identifies turns where the user expresses frustration, distress, or excitement, and checks whether the agent responded appropriately. |
| **Vocabulary Match** | Does the agent mirror the user's language register? Checks whether the agent matches the user's formality level and vocabulary complexity, avoiding talking down or over their head. |

## Quick start

```bash
# Install uv (if you haven't already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up
git clone https://github.com/your-org/voice-agent-evals
cd voice-agent-evals

# Create venv + install all dependencies (including dev)
uv sync

# Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Evaluate a recording
uv run voice-agent-evals evaluate path/to/recording.wav

# With a scenario config for richer scoring
uv run voice-agent-evals evaluate recording.wav --scenario scenarios/restaurant-booking.yaml

# Run tests
uv run pytest

# Start the API server
cd backend && uv run uvicorn api.main:app --reload --port 8000

# Start the frontend
cd frontend && npm install && npm run dev
```


## Two evaluation modes

### Mode 1: Upload & Evaluate
Upload any audio recording. No platform integration needed.

```bash
voice-agent-evals evaluate recording.wav --model base
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


## License

MIT
