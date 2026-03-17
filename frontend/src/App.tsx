import { useState } from 'react'
import NavBar from './components/NavBar'
import UploadView from './components/UploadView'
import ReportView from './components/ReportView'
import HistoryView from './components/HistoryView'
import type { AppView, EvaluationReport } from './types'

export default function App() {
  const [view, setView] = useState<AppView>('upload')
  const [activeReport, setActiveReport] = useState<EvaluationReport | null>(null)

  function showReport(report: EvaluationReport) {
    setActiveReport(report)
    setView('report')
  }

  function goUpload() {
    setActiveReport(null)
    setView('upload')
  }

  return (
    <div className="min-h-screen bg-canvas flex flex-col">
      {/* Grid background */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage:
            'linear-gradient(rgba(79,139,255,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(79,139,255,0.025) 1px, transparent 1px)',
          backgroundSize: '32px 32px',
        }}
      />

      <NavBar currentView={view} onNav={setView} />

      <main className="flex-1 relative z-10">
        {view === 'upload' && <UploadView onReport={showReport} />}
        {view === 'report' && activeReport && (
          <ReportView report={activeReport} onBack={goUpload} />
        )}
        {view === 'history' && <HistoryView onSelectReport={showReport} />}
      </main>

      <footer className="relative z-10 border-t border-border py-4 px-8 flex items-center justify-between">
        <span className="font-mono text-xs text-muted">voice-evals v0.1.0</span>
        <a
          href="https://github.com/your-org/voice-evals"
          target="_blank"
          rel="noopener noreferrer"
          className="font-mono text-xs text-muted hover:text-accent transition-colors"
        >
          MIT License
        </a>
      </footer>
    </div>
  )
}
