"""
CLI for voice-agent-evals.

Usage:
    voice-agent-evals evaluate path/to/recording.wav
    voice-agent-evals evaluate path/to/recording.mp3 --scenario scenarios/booking.yaml
    voice-agent-evals evaluate path/to/recording.wav --model medium --output report.json

    voice-agent-evals inspect path/to/recording.wav
    voice-agent-evals inspect path/to/recording.wav --model small --output trace.json
    voice-agent-evals inspect path/to/recording.wav --json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()  # loads .env from cwd or any parent directory
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="voice-agent-evals",
    help="Evaluate voice AI agent recordings.",
    add_completion=False,
)
console = Console()


@app.command()
def evaluate(
    audio_file: Path = typer.Argument(..., help="Path to audio file (WAV/MP3/OGG/M4A/FLAC)"),
    scenario: Optional[Path] = typer.Option(
        None, "--scenario", "-s", help="Path to scenario YAML config"
    ),
    model: str = typer.Option("base", "--model", "-m", help="Whisper model size"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save JSON report to this path"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show per-metric details"),
    tier: Optional[list[str]] = typer.Option(
        None, "--tier", "-t",
        help="Metric group(s) to run (outcome, technical, quality). Repeatable. Default: all."
    ),
):
    """Evaluate a voice recording and print a scored report."""
    if not audio_file.exists():
        console.print(f"[red]Error: File not found: {audio_file}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Voice Agent Evals[/bold] — evaluating [cyan]{audio_file.name}[/cyan]")

    # Load optional scenario
    scenario_config = None
    if scenario:
        import yaml
        from voice_agent_evals.trace import ScenarioConfig
        with open(scenario) as f:
            data = yaml.safe_load(f)
        scenario_config = ScenarioConfig(**data)
        console.print(f"  Scenario: [yellow]{scenario_config.scenario_id or scenario.name}[/yellow]")

    console.print(f"  Whisper model: [yellow]{model}[/yellow]")
    console.print()

    # Load and process audio
    with console.status("Loading audio..."):
        from voice_agent_evals.ingestion.audio import load_audio, split_channels
        audio = load_audio(str(audio_file))
    console.print(
        f"  [green]✓[/green] Audio loaded — {audio.duration_ms / 1000:.1f}s, "
        f"{audio.n_channels}ch, {audio.sample_rate}Hz"
    )

    # Transcribe
    with console.status(f"Transcribing with Whisper ({model})..."):
        from voice_agent_evals.ingestion.transcribe import (
            WhisperXBackend, merge_and_sort_turns,
            transcribe_stereo, transcribe_with_diarization,
        )
        from voice_agent_evals.trace import (
            AudioInfo, PlatformInfo, Speaker, TimingInfo, Turn, VoiceTrace,
        )

        backend = WhisperXBackend(model_size=model)
        trace = VoiceTrace(
            audio_info=AudioInfo(
                original_file=audio_file.name,
                duration_ms=audio.duration_ms,
                sample_rate=audio.sample_rate,
                channels=audio.n_channels,
                format=audio.format,
            ),
            platform_info=PlatformInfo(platform="upload"),
            scenario=scenario_config,
        )

        if audio.is_stereo:
            user_s, agent_s = split_channels(audio)
            user_result, agent_result = transcribe_stereo(
                user_s, agent_s, audio.sample_rate, backend
            )
            interleaved = merge_and_sort_turns(user_result, agent_result)
            for speaker_label, seg in interleaved:
                trace.add_turn(Turn(
                    speaker=Speaker.USER if speaker_label == "user" else Speaker.AGENT,
                    transcript=seg.text,
                    transcript_confidence=seg.confidence,
                    timing=TimingInfo(speech_start_ms=seg.start_ms, speech_end_ms=seg.end_ms, source="vad"),
                ))
        else:
            diarized, speaker_map = transcribe_with_diarization(audio.mono_mix, audio.sample_rate, backend)
            for seg in diarized.segments:
                spk = speaker_map.get(seg.speaker, "user")
                trace.add_turn(Turn(
                    speaker=Speaker.AGENT if spk == "agent" else Speaker.USER,
                    transcript=seg.text,
                    timing=TimingInfo(speech_start_ms=seg.start_ms, speech_end_ms=seg.end_ms, source="vad"),
                ))

    console.print(f"  [green]✓[/green] Transcription complete — {len(trace.turns)} turns detected")

    # Run evaluation
    with console.status("Running LLM evaluation..."):
        from voice_agent_evals.evaluator import Evaluator, TIER_METRICS
        if tier:
            metric_classes = []
            for t in sorted(set(tier)):
                if t not in TIER_METRICS:
                    valid = sorted(TIER_METRICS)
                    console.print(f"[red]Error: Unknown metric group '{t}'. Valid groups: {valid}[/red]")
                    raise typer.Exit(1)
                metric_classes.extend(TIER_METRICS[t])
        else:
            metric_classes = None  # evaluator uses DEFAULT_METRICS (all tiers)
        evaluator = Evaluator(metrics=metric_classes, embed_trace=False)
        report = evaluator.run(trace)

    console.print(f"  [green]✓[/green] Evaluation complete ({report.duration_ms:.0f}ms)\n")

    # Print summary
    _print_report(report, verbose)

    # Save JSON output
    if output:
        with open(output, "w") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2, default=str)
        console.print(f"\n[dim]Report saved to {output}[/dim]")


def _print_report(report, verbose: bool):
    summary = report.summary
    overall = summary.get("overall_score", 0.0)
    label = summary.get("overall_label", "unknown")

    color = {"excellent": "green", "good": "cyan", "fair": "yellow", "poor": "red"}.get(label, "white")

    console.print(Panel(
        f"Overall Score: [{color}]{overall:.0%}[/{color}]  ({label})\n"
        f"Turns: {summary.get('n_turns', 0)} total  "
        f"({summary.get('n_user_turns', 0)} user, {summary.get('n_agent_turns', 0)} agent)",
        title="[bold]Evaluation Summary[/bold]",
        expand=False,
    ))

    table = Table(title="Metric Scores", show_header=True, header_style="bold")
    table.add_column("Metric", style="dim")
    table.add_column("Score", justify="right")
    table.add_column("Label")
    table.add_column("Reasoning")

    for result in report.results:
        score = result.score.score
        lbl = result.score.label
        c = {"excellent": "green", "completed": "green", "good": "cyan",
             "fair": "yellow", "partial": "yellow", "poor": "red", "failed": "red"}.get(lbl, "white")
        reasoning = result.score.reasoning
        if not verbose and len(reasoning) > 80:
            reasoning = reasoning[:77] + "..."
        table.add_row(
            result.metric_name,
            f"[{c}]{score:.0%}[/{c}]",
            f"[{c}]{lbl}[/{c}]",
            reasoning,
        )

    console.print(table)

    if verbose:
        console.print()
        for result in report.results:
            details = result.score.details
            if details and any(v for v in details.values()):
                console.print(f"[bold]{result.metric_name}[/bold] details:")
                console.print(json.dumps(details, indent=2))
                console.print()


@app.command()
def inspect(
    audio_file: Path = typer.Argument(..., help="Path to audio file (WAV/MP3/OGG/M4A/FLAC)"),
    model: str = typer.Option("base", "--model", "-m", help="Whisper model size"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save trace JSON to this path"),
    table: bool = typer.Option(False, "--table", help="Show transcript table only"),
    json_out: bool = typer.Option(False, "--json", help="Show JSON only"),
    backend: str = typer.Option("whisperx", "--backend", "-b", help="Transcription backend: whisperx, assemblyai"),
    num_speakers: Optional[int] = typer.Option(None, "--speakers", help="Number of speakers hint for diarization (auto-detected if omitted)"),
):
    """Transcribe a recording and print the VoiceTrace. Stops before LLM evaluation."""
    if not audio_file.exists():
        console.print(f"[red]Error: File not found: {audio_file}[/red]")
        raise typer.Exit(1)

    if backend not in ("whisperx", "assemblyai"):
        console.print(f"[red]Error: Unknown backend '{backend}'. Choose: whisperx, assemblyai[/red]")
        raise typer.Exit(1)

    # If neither flag set, show both. If one or both set, show only those requested.
    show_table = table or (not table and not json_out)
    show_json = json_out or (not table and not json_out)

    if show_table:
        console.print(f"\n[bold]Voice Agent Evals — Inspect[/bold] — [cyan]{audio_file.name}[/cyan]")
        console.print(f"  Backend: [yellow]{backend}[/yellow]" + (f"  Whisper model: [yellow]{model}[/yellow]" if backend == "whisperx" else "") + "\n")

    # Load audio
    with console.status("Loading audio..."):
        from voice_agent_evals.ingestion.audio import load_audio, split_channels
        audio = load_audio(str(audio_file))

    if show_table:
        console.print(
            f"  [green]✓[/green] Audio loaded — {audio.duration_ms / 1000:.1f}s, "
            f"{audio.n_channels}ch, {audio.sample_rate}Hz"
        )

    from voice_agent_evals.trace import (
        AudioInfo, PlatformInfo, Speaker, TimingInfo, Turn, VoiceTrace,
    )

    trace = VoiceTrace(
        audio_info=AudioInfo(
            original_file=audio_file.name,
            duration_ms=audio.duration_ms,
            sample_rate=audio.sample_rate,
            channels=audio.n_channels,
            format=audio.format,
            user_channel=0 if audio.is_stereo else None,
            agent_channel=1 if audio.is_stereo else None,
        ),
        platform_info=PlatformInfo(platform="upload"),
    )

    if backend == "assemblyai":
        # --- AssemblyAI path: cloud API, works directly with the file ---
        if show_table:
            console.print("  Sending audio to AssemblyAI for transcription + diarization...")
        try:
            from voice_agent_evals.ingestion.transcribe import AssemblyAIBackend, transcribe_with_assemblyai
            aai_backend = AssemblyAIBackend(num_speakers=num_speakers)
            with console.status("Transcribing + diarizing with AssemblyAI..."):
                diarized, speaker_map = transcribe_with_assemblyai(str(audio_file), aai_backend)
        except RuntimeError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        if show_table:
            console.print(f"  [green]✓[/green] Detected {diarized.num_speakers} speaker(s): {speaker_map}")
        for seg in diarized.segments:
            spk = speaker_map.get(seg.speaker, "user")
            trace.add_turn(Turn(
                speaker=Speaker.AGENT if spk == "agent" else Speaker.USER,
                transcript=seg.text,
                transcript_confidence=seg.confidence,
                timing=TimingInfo(speech_start_ms=seg.start_ms, speech_end_ms=seg.end_ms, source="assemblyai"),
            ))

    else:
        # --- WhisperX path: local model ---
        if show_table:
            console.print(f"  Loading WhisperX '{model}' model (may download on first use)...")
        try:
            from voice_agent_evals.ingestion.transcribe import (
                WhisperXBackend, merge_and_sort_turns,
                transcribe_stereo, transcribe_with_diarization,
            )
            wx_backend = WhisperXBackend(model_size=model, num_speakers=num_speakers)
            wx_backend._get_model()
        except RuntimeError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        if show_table:
            console.print(f"  [green]✓[/green] WhisperX '{model}' model ready")

        if audio.is_stereo:
            with console.status("Transcribing stereo channels with WhisperX..."):
                if show_table:
                    console.print("  Transcribing user channel (left)...")
                user_s, agent_s = split_channels(audio)
                user_result, agent_result = transcribe_stereo(user_s, agent_s, audio.sample_rate, wx_backend)
                if show_table:
                    console.print("  Transcribing agent channel (right)...")
            interleaved = merge_and_sort_turns(user_result, agent_result)
            for speaker_label, seg in interleaved:
                trace.add_turn(Turn(
                    speaker=Speaker.USER if speaker_label == "user" else Speaker.AGENT,
                    transcript=seg.text,
                    transcript_confidence=seg.confidence,
                    timing=TimingInfo(speech_start_ms=seg.start_ms, speech_end_ms=seg.end_ms, source="vad"),
                ))
        else:
            with console.status("Transcribing + diarizing with WhisperX..."):
                if show_table:
                    console.print("  Mono audio — running transcription and speaker diarization...")
                try:
                    diarized, speaker_map = transcribe_with_diarization(
                        audio.mono_mix, audio.sample_rate, wx_backend
                    )
                except RuntimeError as e:
                    console.print(f"[red]Error: {e}[/red]")
                    raise typer.Exit(1)
            if show_table:
                console.print(f"  [green]✓[/green] Detected {diarized.num_speakers} speaker(s): {speaker_map}")
            if show_table and diarized.overlap_regions_ms:
                console.print(f"  [yellow]⚡[/yellow] Detected {len(diarized.overlap_regions_ms)} overlap region(s)")
            for seg in diarized.segments:
                spk = speaker_map.get(seg.speaker, "user")
                trace.add_turn(Turn(
                    speaker=Speaker.AGENT if spk == "agent" else Speaker.USER,
                    transcript=seg.text,
                    transcript_confidence=seg.confidence,
                    timing=TimingInfo(speech_start_ms=seg.start_ms, speech_end_ms=seg.end_ms, source="vad"),
                    platform_metadata={"is_overlap": seg.is_overlap},
                ))

    if show_table:
        console.print(f"  [green]✓[/green] Transcription complete — {len(trace.turns)} turns detected\n")

        # Print transcript table
        transcript_table = Table(title="Transcript", show_header=True, header_style="bold")
        transcript_table.add_column("#", style="dim", width=4)
        transcript_table.add_column("Speaker", width=8)
        transcript_table.add_column("Transcript")
        transcript_table.add_column("Start", justify="right", width=10)
        transcript_table.add_column("End", justify="right", width=10)
        transcript_table.add_column("Confidence", justify="right", width=10)
        transcript_table.add_column("Overlap", justify="center", width=9)

        for i, turn in enumerate(trace.turns):
            color = "cyan" if turn.speaker == Speaker.USER else "green"
            start = f"{turn.timing.speech_start_ms / 1000:.2f}s" if turn.timing and turn.timing.speech_start_ms is not None else "—"
            end = f"{turn.timing.speech_end_ms / 1000:.2f}s" if turn.timing and turn.timing.speech_end_ms is not None else "—"
            conf = f"{turn.transcript_confidence:.0%}" if turn.transcript_confidence is not None else "—"
            overlap = "[yellow]yes[/yellow]" if turn.platform_metadata.get("is_overlap") else "—"
            transcript_table.add_row(
                str(i + 1),
                f"[{color}]{turn.speaker.value}[/{color}]",
                turn.transcript or "",
                start,
                end,
                conf,
                overlap,
            )

        console.print(transcript_table)
        console.print()

    # Print JSON
    if show_json:
        trace_json = json.dumps(trace.model_dump(mode="json"), indent=2, default=str)
        if show_table:
            console.print("[dim]--- Full VoiceTrace JSON ---[/dim]")
        print(trace_json)

    # Save to file
    if output:
        trace_json = json.dumps(trace.model_dump(mode="json"), indent=2, default=str)
        with open(output, "w") as f:
            f.write(trace_json)
        if show_table:
            console.print(f"\n[dim]Trace saved to {output}[/dim]")


if __name__ == "__main__":
    app()
