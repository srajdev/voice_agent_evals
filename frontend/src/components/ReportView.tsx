import type { EvaluationReport } from '../types'
import ScoreGauge from './ScoreGauge'
import MetricCard from './MetricCard'
import TranscriptTable from './TranscriptTable'

interface Props {
  report: EvaluationReport
  onBack: () => void
}

export default function ReportView({ report, onBack }: Props) {
  const { summary } = report
  const turns = report.trace?.turns ?? []
  const audioInfo = report.trace?.audio_info

  return (
    <div className="max-w-5xl mx-auto px-6 py-12 animate-slide-up">
      {/* Back button + meta */}
      <div className="flex items-center justify-between mb-10">
        <button
          onClick={onBack}
          className="flex items-center gap-2 font-mono text-xs text-muted hover:text-accent transition-colors group"
        >
          <span className="group-hover:-translate-x-0.5 transition-transform">←</span>
          Evaluate Another
        </button>

        <div className="flex items-center gap-3">
          <div className="font-mono text-[10px] text-muted">
            Report ID:{' '}
            <span className="text-white/60">{report.report_id.slice(0, 8)}…</span>
          </div>
          <div className="font-mono text-[10px] text-muted">
            {new Date(report.evaluated_at).toLocaleString()}
          </div>
        </div>
      </div>

      {/* Hero: score + stats */}
      <div className="bg-panel border border-border rounded-2xl p-8 mb-8 relative overflow-hidden">
        {/* Background gradient */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: `radial-gradient(600px circle at 80% 50%, rgba(79,139,255,0.04), transparent)`,
          }}
        />

        <div className="relative flex flex-col md:flex-row items-center md:items-start gap-8">
          {/* Gauge */}
          <div className="shrink-0">
            <ScoreGauge
              score={summary.overall_score}
              label={summary.overall_label}
              size={200}
            />
          </div>

          {/* Stats grid */}
          <div className="flex-1 w-full">
            <h2
              className="font-display text-2xl text-white mb-6"
              style={{ fontWeight: 700 }}
            >
              Evaluation Report
            </h2>

            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <Stat label="Total Turns" value={summary.n_turns} />
              <Stat label="User Turns" value={summary.n_user_turns} />
              <Stat label="Agent Turns" value={summary.n_agent_turns} />
              {summary.duration_ms && (
                <Stat
                  label="Duration"
                  value={`${(summary.duration_ms / 1000).toFixed(1)}s`}
                />
              )}
              <Stat label="Platform" value={summary.platform} mono />
              <Stat
                label="Eval Time"
                value={`${(report.duration_ms / 1000).toFixed(1)}s`}
              />
              {audioInfo && (
                <>
                  {audioInfo.format && <Stat label="Format" value={audioInfo.format.toUpperCase()} mono />}
                  {audioInfo.sample_rate && <Stat label="Sample Rate" value={`${audioInfo.sample_rate / 1000}kHz`} mono />}
                  {audioInfo.channels && (
                    <Stat label="Channels" value={audioInfo.channels === 2 ? 'Stereo' : 'Mono'} mono />
                  )}
                </>
              )}
              {summary.scenario && (
                <Stat label="Scenario" value={summary.scenario} mono />
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Metric cards */}
      <section className="mb-8">
        <SectionHeader title="Metric Scores" subtitle="LLM-judge evaluation across 4 dimensions" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {report.results.map(result => (
            <MetricCard key={result.metric_name} result={result} />
          ))}
        </div>
      </section>

      {/* Transcript */}
      {turns.length > 0 && (
        <section>
          <SectionHeader
            title="Conversation Transcript"
            subtitle={`${turns.length} turns · ${turns.filter(t => t.speaker === 'user').length} user · ${turns.filter(t => t.speaker === 'agent').length} agent`}
          />
          <TranscriptTable turns={turns} />
        </section>
      )}

      {/* Raw JSON toggle */}
      <RawJsonSection report={report} />
    </div>
  )
}

function Stat({
  label,
  value,
  mono = false,
}: {
  label: string
  value: string | number
  mono?: boolean
}) {
  return (
    <div className="bg-surface border border-border rounded-lg px-3 py-2.5">
      <p className="font-mono text-[10px] text-muted uppercase tracking-wide mb-1">{label}</p>
      <p className={`text-sm text-white ${mono ? 'font-mono' : 'font-medium'}`}>{value}</p>
    </div>
  )
}

function SectionHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div className="mb-4 flex items-baseline justify-between">
      <h3 className="font-display text-lg text-white" style={{ fontWeight: 600 }}>
        {title}
      </h3>
      {subtitle && <p className="font-mono text-xs text-muted">{subtitle}</p>}
    </div>
  )
}

function RawJsonSection({ report }: { report: EvaluationReport }) {
  const [open, setOpen] = React.useState(false)

  return (
    <div className="mt-8">
      <button
        onClick={() => setOpen(o => !o)}
        className="flex items-center gap-2 font-mono text-xs text-muted hover:text-accent transition-colors"
      >
        <span className={`transition-transform ${open ? 'rotate-90' : ''}`}>▶</span>
        {open ? 'Hide' : 'Show'} raw JSON
      </button>
      {open && (
        <pre className="mt-3 bg-panel border border-border rounded-xl p-4 text-xs font-mono text-white/60 overflow-auto max-h-96 leading-relaxed">
          {JSON.stringify(report, null, 2)}
        </pre>
      )}
    </div>
  )
}

// Inline React import needed for useState in sub-component
import React from 'react'
