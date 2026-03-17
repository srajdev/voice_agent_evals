import type { Turn } from '../types'

interface Props {
  turns: Turn[]
}

export default function TranscriptTable({ turns }: Props) {
  if (!turns || turns.length === 0) {
    return (
      <p className="font-mono text-xs text-muted py-6 text-center">
        No transcript available.
      </p>
    )
  }

  return (
    <div className="overflow-hidden rounded-xl border border-border">
      <table className="w-full">
        <thead>
          <tr className="border-b border-border bg-panel/80">
            <th className="px-4 py-3 text-left font-mono text-[10px] text-muted uppercase tracking-widest w-12">#</th>
            <th className="px-4 py-3 text-left font-mono text-[10px] text-muted uppercase tracking-widest w-24">Speaker</th>
            <th className="px-4 py-3 text-left font-mono text-[10px] text-muted uppercase tracking-widest">Transcript</th>
            <th className="px-4 py-3 text-right font-mono text-[10px] text-muted uppercase tracking-widest w-28 hidden md:table-cell">Timing</th>
          </tr>
        </thead>
        <tbody>
          {turns.map((turn, i) => (
            <tr
              key={turn.turn_id}
              className={`
                border-b border-border/50 last:border-0 transition-colors
                ${turn.speaker === 'user' ? 'hover:bg-accent/3' : 'hover:bg-teal/3'}
              `}
            >
              <td className="px-4 py-3 font-mono text-xs text-muted">{i + 1}</td>
              <td className="px-4 py-3">
                <SpeakerBadge speaker={turn.speaker} />
              </td>
              <td className="px-4 py-3 font-mono text-xs text-white/80 leading-relaxed">
                {turn.transcript ?? (
                  <span className="text-muted italic">[no transcript]</span>
                )}
                {turn.transcript_confidence !== undefined && (
                  <span className="ml-2 text-muted/50 text-[10px]">
                    {Math.round(turn.transcript_confidence * 100)}%
                  </span>
                )}
              </td>
              <td className="px-4 py-3 text-right font-mono text-[10px] text-muted hidden md:table-cell">
                {turn.timing?.speech_start_ms !== undefined ? (
                  <span>
                    {(turn.timing.speech_start_ms / 1000).toFixed(1)}s
                    {turn.timing.speech_end_ms !== undefined && (
                      <> → {(turn.timing.speech_end_ms / 1000).toFixed(1)}s</>
                    )}
                  </span>
                ) : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function SpeakerBadge({ speaker }: { speaker: string }) {
  if (speaker === 'user') {
    return (
      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-accent/10 border border-accent/20 font-mono text-[10px] text-accent uppercase tracking-wide">
        <span className="w-1.5 h-1.5 rounded-full bg-accent" />
        User
      </span>
    )
  }
  if (speaker === 'agent') {
    return (
      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-teal/10 border border-teal/20 font-mono text-[10px] text-teal uppercase tracking-wide">
        <span className="w-1.5 h-1.5 rounded-full bg-teal" />
        Agent
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-subtle font-mono text-[10px] text-muted uppercase tracking-wide">
      {speaker}
    </span>
  )
}
