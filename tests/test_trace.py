"""Tests for VoiceTrace schema."""

import pytest
from voice_agent_evals.trace import (
    AudioInfo,
    PlatformInfo,
    ScenarioConfig,
    Speaker,
    TimingInfo,
    Turn,
    VoiceTrace,
)


def make_trace() -> VoiceTrace:
    trace = VoiceTrace()
    trace.add_turn(Turn(speaker=Speaker.USER, transcript="Hi, I want to book a table."))
    trace.add_turn(Turn(speaker=Speaker.AGENT, transcript="Sure! For how many people?"))
    trace.add_turn(Turn(speaker=Speaker.USER, transcript="For two please."))
    trace.add_turn(Turn(speaker=Speaker.AGENT, transcript="Got it. What time?"))
    return trace


def test_trace_turn_counts():
    trace = make_trace()
    assert len(trace.turns) == 4
    assert len(trace.user_turns) == 2
    assert len(trace.agent_turns) == 2


def test_trace_full_transcript():
    trace = make_trace()
    transcript = trace.full_transcript
    assert "USER:" in transcript
    assert "AGENT:" in transcript
    assert "book a table" in transcript


def test_trace_turn_pairs():
    trace = make_trace()
    pairs = trace.get_turn_pairs()
    assert len(pairs) == 2
    for user_turn, agent_turn in pairs:
        assert user_turn.speaker == Speaker.USER
        assert agent_turn.speaker == Speaker.AGENT


def test_audio_info():
    trace = VoiceTrace(
        audio_info=AudioInfo(
            duration_ms=30000,
            sample_rate=16000,
            channels=2,
            format="wav",
        )
    )
    assert trace.duration_ms == 30000


def test_scenario_config():
    trace = VoiceTrace(
        scenario=ScenarioConfig(
            scenario_id="test-001",
            expected_task="Book a table",
            expected_intents=["request_reservation", "provide_party_size"],
        )
    )
    assert trace.scenario is not None
    assert len(trace.scenario.expected_intents) == 2


def test_empty_trace():
    trace = VoiceTrace()
    assert trace.user_turns == []
    assert trace.agent_turns == []
    assert trace.duration_ms is None
    assert trace.full_transcript == ""


def test_turn_ids_unique():
    trace = make_trace()
    ids = [t.turn_id for t in trace.turns]
    assert len(ids) == len(set(ids))
