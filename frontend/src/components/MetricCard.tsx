import type { MetricResult } from '../types'

const METRIC_META: Record<string, { title: string; description: string; icon: string }> = {
  conversation_quality: {
    title: 'Conversation Quality',
    description: 'Voice-first communication principles',
    icon: '◈',
  },
  coherence: {
    title: 'Multi-turn Coherence',
    description: 'Context retention across turns',
    icon: '⟳',
  },
  intent_accuracy: {
    title: 'Intent Accuracy',
    description: 'Correct interpretation of user requests',
    icon: '◎',
  },
  task_completion: {
    title: 'Task Completion',
    description: 'End-to-end goal achievement',
    icon: '◉',
  },
}

function scoreColor(score: number): string {
  if (score >= 0.85) return '#00d4aa'
  if (score >= 0.70) return '#4f8bff'
  if (score >= 0.50) return '#f5a623'
  return '#ff4f6a'
}

interface Props {
  result: MetricResult
}

export default function MetricCard({ result }: Props) {
  const meta = METRIC_META[result.metric_name] ?? {
    title: result.metric_name,
    description: '',
    icon: '◇',
  }
  const color = scoreColor(result.score.score)
  const pct = Math.round(result.score.score * 100)

  return (
    <div
      className="bg-panel border border-border rounded-xl p-5 flex flex-col gap-3 hover:border-accent/20 transition-colors group relative overflow-hidden"
    >
      {/* Background score bar */}
      <div
        className="absolute bottom-0 left-0 h-0.5 transition-all duration-1000"
        style={{ width: `${pct}%`, background: color, opacity: 0.6 }}
      />

      {/* Header row */}
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-muted font-mono text-sm">{meta.icon}</span>
            <h3 className="font-display text-sm font-600 text-white" style={{ fontWeight: 600 }}>
              {meta.title}
            </h3>
          </div>
          <p className="font-mono text-xs text-muted mt-0.5">{meta.description}</p>
        </div>

        {/* Score pill */}
        <div className="flex flex-col items-end shrink-0">
          <span
            className="font-display text-2xl leading-none"
            style={{ fontWeight: 800, color }}
          >
            {pct}
          </span>
          <span className="font-mono text-[10px] text-muted">/ 100</span>
        </div>
      </div>

      {/* Label badge */}
      <span
        className="self-start px-2 py-0.5 rounded-md text-[10px] font-mono uppercase tracking-widest"
        style={{ background: `${color}15`, color, border: `1px solid ${color}25` }}
      >
        {result.score.label}
      </span>

      {/* Reasoning */}
      <p className="text-xs text-muted leading-relaxed line-clamp-3">
        {result.score.reasoning || 'No reasoning provided.'}
      </p>

      {/* Details (issues/strengths) */}
      <Details result={result} color={color} />
    </div>
  )
}

function Details({ result, color }: { result: MetricResult; color: string }) {
  const issues = (result.score.details?.issues as string[]) ?? []
  const strengths = (result.score.details?.strengths as string[]) ?? []
  const completedSteps = (result.score.details?.completed_steps as string[]) ?? []
  const missingSteps = (result.score.details?.missing_steps as string[]) ?? []
  const failures = (result.score.details?.coherence_failures as string[]) ?? []

  const hasContent =
    issues.length > 0 ||
    strengths.length > 0 ||
    completedSteps.length > 0 ||
    missingSteps.length > 0 ||
    failures.length > 0

  if (!hasContent) return null

  return (
    <div className="mt-1 space-y-2">
      {strengths.map((s, i) => (
        <div key={i} className="flex gap-2 text-[11px] text-teal">
          <span className="shrink-0 mt-0.5">✓</span>
          <span>{s}</span>
        </div>
      ))}
      {completedSteps.map((s, i) => (
        <div key={i} className="flex gap-2 text-[11px] text-teal">
          <span className="shrink-0 mt-0.5">✓</span>
          <span>{s}</span>
        </div>
      ))}
      {issues.map((s, i) => (
        <div key={i} className="flex gap-2 text-[11px] text-amber">
          <span className="shrink-0 mt-0.5">⚠</span>
          <span>{s}</span>
        </div>
      ))}
      {missingSteps.map((s, i) => (
        <div key={i} className="flex gap-2 text-[11px] text-amber">
          <span className="shrink-0 mt-0.5">⚠</span>
          <span>{s}</span>
        </div>
      ))}
      {failures.map((s, i) => (
        <div key={i} className="flex gap-2 text-[11px] text-red">
          <span className="shrink-0 mt-0.5">✗</span>
          <span>{s}</span>
        </div>
      ))}
    </div>
  )
}
