"""
CLI for voice-evals.

Usage:
    voice-evals evaluate path/to/recording.wav
    voice-evals evaluate path/to/recording.mp3 --scenario scenarios/booking.yaml
    voice-evals evaluate path/to/recording.wav --model medium --output report.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()  # loads .env from cwd or any parent directory
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="voice-evals",
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
):
    """Evaluate a voice recording and print a scored report."""
    if not audio_file.exists():
        console.print(f"[red]Error: File not found: {audio_file}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Voice Evals[/bold] — evaluating [cyan]{audio_file.name}[/cyan]")

    # Load optional scenario
    scenario_config = None
    if scenario:
        import yaml
        from voice_evals.trace import ScenarioConfig
        with open(scenario) as f:
            data = yaml.safe_load(f)
        scenario_config = ScenarioConfig(**data)
        console.print(f"  Scenario: [yellow]{scenario_config.scenario_id or scenario.name}[/yellow]")

    console.print(f"  Whisper model: [yellow]{model}[/yellow]")
    console.print()

    # Load and process audio
    with console.status("Loading audio..."):
        from voice_evals.ingestion.audio import load_audio, split_channels
        audio = load_audio(str(audio_file))
    console.print(
        f"  [green]✓[/green] Audio loaded — {audio.duration_ms / 1000:.1f}s, "
        f"{audio.n_channels}ch, {audio.sample_rate}Hz"
    )

    # Transcribe
    with console.status(f"Transcribing with Whisper ({model})..."):
        from voice_evals.ingestion.transcribe import (
            WhisperBackend, merge_and_sort_turns,
            transcribe_stereo, transcribe_mono,
        )
        from voice_evals.trace import (
            AudioInfo, PlatformInfo, Speaker, TimingInfo, Turn, VoiceTrace,
        )

        backend = WhisperBackend(model_size=model)
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
            result = transcribe_mono(audio.mono_mix, audio.sample_rate, backend)
            for i, seg in enumerate(result.segments):
                trace.add_turn(Turn(
                    speaker=Speaker.USER if i % 2 == 0 else Speaker.AGENT,
                    transcript=seg.text,
                    timing=TimingInfo(speech_start_ms=seg.start_ms, speech_end_ms=seg.end_ms, source="estimated"),
                ))

    console.print(f"  [green]✓[/green] Transcription complete — {len(trace.turns)} turns detected")

    # Run evaluation
    with console.status("Running LLM evaluation..."):
        from voice_evals.evaluator import Evaluator
        evaluator = Evaluator(embed_trace=False)
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


if __name__ == "__main__":
    app()
