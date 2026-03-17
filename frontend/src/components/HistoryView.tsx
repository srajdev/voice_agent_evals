import { useEffect, useState } from 'react'
import type { EvaluationReport, ReportListItem, ReportListResponse } from '../types'

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string) || ''

function scoreColor(score: number): string {
  if (score >= 0.85) return '#00d4aa'
  if (score >= 0.70) return '#4f8bff'
  if (score >= 0.50) return '#f5a623'
  return '#ff4f6a'
}

interface Props {
  onSelectReport: (report: EvaluationReport) => void
}

export default function HistoryView({ onSelectReport }: Props) {
  const [items, setItems] = useState<ReportListItem[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [loadingId, setLoadingId] = useState<string | null>(null)

  useEffect(() => {
    fetchReports()
  }, [])

  async function fetchReports() {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/api/v1/reports?limit=50`)
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      const data: ReportListResponse = await res.json()
      setItems(data.reports)
      setTotal(data.total)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  async function handleRowClick(reportId: string) {
    setLoadingId(reportId)
    try {
      const res = await fetch(`${API_BASE}/api/v1/reports/${reportId}`)
      if (!res.ok) throw new Error(`${res.status}`)
      const report: EvaluationReport = await res.json()
      onSelectReport(report)
    } catch (e) {
      // silent — keep on history page
    } finally {
      setLoadingId(null)
    }
  }

  return (
    <div className="max-w-5xl mx-auto px-6 py-12 animate-slide-up">
      <div className="flex items-baseline justify-between mb-8">
        <div>
          <p className="font-mono text-xs text-accent mb-2 tracking-widest uppercase">
            Evaluation History
          </p>
          <h1
            className="font-display text-3xl text-white"
            style={{ fontWeight: 700 }}
          >
            Past Reports
          </h1>
        </div>
        <button
          onClick={fetchReports}
          className="font-mono text-xs text-muted hover:text-accent transition-colors flex items-center gap-1.5"
        >
          <span className={loading ? 'animate-spin' : ''}>↻</span>
          Refresh
        </button>
      </div>

      {loading && (
        <div className="flex justify-center py-20">
          <LoadingDots />
        </div>
      )}

      {error && (
        <div className="bg-red/10 border border-red/30 rounded-xl px-5 py-4">
          <p className="font-mono text-xs text-red">{error}</p>
        </div>
      )}

      {!loading && !error && items.length === 0 && (
        <div className="text-center py-20">
          <p className="font-display text-xl text-muted" style={{ fontWeight: 600 }}>
            No evaluations yet
          </p>
          <p className="font-mono text-xs text-muted/50 mt-2">
            Upload a recording to generate your first report
          </p>
        </div>
      )}

      {!loading && items.length > 0 && (
        <>
          <div className="font-mono text-xs text-muted mb-4">
            {total} report{total !== 1 ? 's' : ''} total
          </div>

          <div className="overflow-hidden rounded-xl border border-border">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border bg-panel/80">
                  <th className="px-5 py-3 text-left font-mono text-[10px] text-muted uppercase tracking-widest">
                    Report ID
                  </th>
                  <th className="px-5 py-3 text-left font-mono text-[10px] text-muted uppercase tracking-widest hidden md:table-cell">
                    Date
                  </th>
                  <th className="px-5 py-3 text-right font-mono text-[10px] text-muted uppercase tracking-widest">
                    Score
                  </th>
                  <th className="px-5 py-3 text-left font-mono text-[10px] text-muted uppercase tracking-widest">
                    Label
                  </th>
                  <th className="px-5 py-3 w-8" />
                </tr>
              </thead>
              <tbody>
                {items.map(item => {
                  const color = scoreColor(item.overall_score)
                  const isLoading = loadingId === item.report_id
                  return (
                    <tr
                      key={item.report_id}
                      onClick={() => handleRowClick(item.report_id)}
                      className="border-b border-border/50 last:border-0 hover:bg-accent/3 cursor-pointer transition-colors group"
                    >
                      <td className="px-5 py-4 font-mono text-xs text-white/70">
                        {item.report_id.slice(0, 8)}
                        <span className="text-muted">…</span>
                      </td>
                      <td className="px-5 py-4 font-mono text-xs text-muted hidden md:table-cell">
                        {new Date(item.evaluated_at).toLocaleString()}
                      </td>
                      <td className="px-5 py-4 text-right">
                        <span
                          className="font-display text-lg leading-none"
                          style={{ fontWeight: 800, color }}
                        >
                          {Math.round(item.overall_score * 100)}
                        </span>
                        <span className="font-mono text-[10px] text-muted ml-0.5">%</span>
                      </td>
                      <td className="px-5 py-4">
                        <span
                          className="inline-block px-2 py-0.5 rounded-md text-[10px] font-mono uppercase tracking-widest"
                          style={{
                            background: `${color}15`,
                            color,
                            border: `1px solid ${color}25`,
                          }}
                        >
                          {item.overall_label}
                        </span>
                      </td>
                      <td className="px-5 py-4 text-right">
                        {isLoading ? (
                          <span className="text-muted font-mono text-xs animate-spin-slow inline-block">⟳</span>
                        ) : (
                          <span className="text-muted group-hover:text-accent transition-colors font-mono text-xs">
                            →
                          </span>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}

function LoadingDots() {
  return (
    <div className="flex items-center gap-1.5">
      {[0, 1, 2].map(i => (
        <div
          key={i}
          className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse"
          style={{ animationDelay: `${i * 0.2}s` }}
        />
      ))}
    </div>
  )
}
