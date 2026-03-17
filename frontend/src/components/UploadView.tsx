import { useState, useRef, DragEvent, ChangeEvent } from 'react'
import type { EvaluationReport } from '../types'

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string) || ''

const STEPS = [
  'Uploading audio…',
  'Loading model…',
  'Transcribing…',
  'Running LLM evaluation…',
  'Finalizing report…',
]

interface Props {
  onReport: (report: EvaluationReport) => void
}

export default function UploadView({ onReport }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)
  const [scenario, setScenario] = useState('')
  const [model, setModel] = useState('base')
  const [loading, setLoading] = useState(false)
  const [stepIdx, setStepIdx] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const stepTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  function handleDrop(e: DragEvent) {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) setFile(f)
  }

  function handleFileChange(e: ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (f) setFile(f)
  }

  async function handleEvaluate() {
    if (!file) return
    setError(null)
    setLoading(true)
    setStepIdx(0)

    // Rotate status text while waiting
    stepTimerRef.current = setInterval(() => {
      setStepIdx(prev => (prev + 1 < STEPS.length ? prev + 1 : prev))
    }, 3500)

    try {
      const form = new FormData()
      form.append('audio', file)
      if (scenario.trim()) form.append('scenario_yaml', scenario.trim())
      form.append('whisper_model', model)

      const res = await fetch(`${API_BASE}/api/v1/evaluate`, {
        method: 'POST',
        body: form,
      })

      if (!res.ok) {
        const text = await res.text()
        throw new Error(`Server error ${res.status}: ${text}`)
      }

      const report: EvaluationReport = await res.json()
      onReport(report)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
      if (stepTimerRef.current) clearInterval(stepTimerRef.current)
    }
  }

  if (loading) {
    return <LoadingScreen stepIdx={stepIdx} fileName={file?.name ?? ''} />
  }

  return (
    <div className="max-w-3xl mx-auto px-6 py-16 animate-slide-up">
      {/* Header */}
      <div className="mb-12">
        <p className="font-mono text-xs text-accent mb-3 tracking-widest uppercase">
          Voice AI Evaluation Framework
        </p>
        <h1 className="font-display text-4xl font-800 text-white leading-tight" style={{ fontWeight: 800 }}>
          Evaluate a<br />
          <span className="text-accent text-glow-accent">voice recording</span>
        </h1>
        <p className="mt-4 text-muted text-sm leading-relaxed max-w-md">
          Upload any voice AI call recording. We'll transcribe, analyse conversation
          quality, coherence, intent accuracy and task completion — then score it.
        </p>
      </div>

      {/* Drop zone */}
      <div
        onDrop={handleDrop}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onClick={() => inputRef.current?.click()}
        className={`
          relative cursor-pointer rounded-xl border-2 transition-all duration-200 overflow-hidden
          ${dragging
            ? 'border-accent glow-accent bg-accent/5'
            : file
            ? 'border-accent/40 bg-accent/5'
            : 'border-border bg-panel hover:border-accent/30 hover:bg-panel/80'
          }
        `}
      >
        {/* Corner decorations */}
        <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-accent/40 rounded-tl-lg" />
        <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-accent/40 rounded-tr-lg" />
        <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-accent/40 rounded-bl-lg" />
        <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-accent/40 rounded-br-lg" />

        <div className="py-12 px-8 flex flex-col items-center gap-4">
          <input
            ref={inputRef}
            type="file"
            accept=".wav,.mp3,.ogg,.m4a,.flac"
            className="hidden"
            onChange={handleFileChange}
          />

          {file ? (
            <>
              <AudioFileIcon />
              <div className="text-center">
                <p className="font-mono text-sm text-white">{file.name}</p>
                <p className="font-mono text-xs text-muted mt-1">
                  {(file.size / (1024 * 1024)).toFixed(2)} MB
                </p>
              </div>
              <p className="text-xs text-accent/60">Click to change file</p>
            </>
          ) : (
            <>
              <UploadIcon dragging={dragging} />
              <div className="text-center">
                <p className="text-sm text-white font-medium">
                  {dragging ? 'Drop it here' : 'Drop audio file or click to browse'}
                </p>
                <p className="text-xs text-muted mt-1">
                  WAV · MP3 · OGG · M4A · FLAC — up to 100MB
                </p>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Options row */}
      <div className="mt-6 grid grid-cols-2 gap-4">
        {/* Whisper model */}
        <div>
          <label className="block font-mono text-xs text-muted mb-2 uppercase tracking-wide">
            Whisper Model
          </label>
          <select
            value={model}
            onChange={e => setModel(e.target.value)}
            className="w-full bg-panel border border-border text-white text-sm rounded-lg px-3 py-2.5 font-mono appearance-none focus:outline-none focus:border-accent/50 transition-colors"
          >
            <option value="tiny">tiny — fastest</option>
            <option value="base">base — recommended</option>
            <option value="small">small — better accuracy</option>
            <option value="medium">medium — production quality</option>
          </select>
        </div>

        {/* Stereo hint */}
        <div className="flex items-end">
          <div className="bg-panel border border-border rounded-lg px-4 py-2.5 text-xs text-muted font-mono leading-relaxed w-full">
            <span className="text-accent/70">tip:</span> stereo audio gives per-speaker metrics (left = user, right = agent)
          </div>
        </div>
      </div>

      {/* Scenario YAML */}
      <div className="mt-6">
        <label className="block font-mono text-xs text-muted mb-2 uppercase tracking-wide">
          Scenario Config{' '}
          <span className="text-muted/50 normal-case">(optional YAML)</span>
        </label>
        <textarea
          value={scenario}
          onChange={e => setScenario(e.target.value)}
          placeholder={`# Example\nscenario_id: booking-test\nexpected_task: Book a dinner reservation for 2 at 7pm\ncompletion_criteria: Agent confirms reservation date, time, and party size\nexpected_intents:\n  - request_reservation\n  - provide_party_size`}
          rows={7}
          className="w-full bg-panel border border-border text-white text-xs rounded-lg px-4 py-3 font-mono placeholder:text-muted/40 focus:outline-none focus:border-accent/50 transition-colors resize-none leading-relaxed"
        />
      </div>

      {/* Error */}
      {error && (
        <div className="mt-4 bg-red/10 border border-red/30 rounded-lg px-4 py-3">
          <p className="font-mono text-xs text-red">{error}</p>
        </div>
      )}

      {/* Submit */}
      <button
        disabled={!file}
        onClick={handleEvaluate}
        className={`
          mt-6 w-full py-3.5 rounded-xl font-display font-600 text-sm tracking-wide transition-all duration-200
          ${file
            ? 'bg-accent text-canvas hover:bg-accent/90 glow-accent hover:scale-[1.01] active:scale-[0.99]'
            : 'bg-subtle text-muted cursor-not-allowed'
          }
        `}
        style={{ fontWeight: 600 }}
      >
        Run Evaluation →
      </button>
    </div>
  )
}

function UploadIcon({ dragging }: { dragging: boolean }) {
  return (
    <div className={`w-14 h-14 rounded-xl flex items-center justify-center transition-all ${dragging ? 'bg-accent/20' : 'bg-subtle'}`}>
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className={`w-7 h-7 ${dragging ? 'text-accent' : 'text-muted'} transition-colors`}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
      </svg>
    </div>
  )
}

function AudioFileIcon() {
  return (
    <div className="w-14 h-14 rounded-xl flex items-center justify-center bg-accent/15">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-7 h-7 text-accent">
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 11-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303m0 0v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 01-.99-3.467l2.31-.66A2.25 2.25 0 009 15.553z" />
      </svg>
    </div>
  )
}

function LoadingScreen({ stepIdx, fileName }: { stepIdx: number; fileName: string }) {
  const bars = [0.3, 0.6, 1, 0.7, 0.4, 0.9, 0.5, 0.8, 0.35, 0.65, 1, 0.45, 0.75, 0.55, 0.9]
  return (
    <div className="flex flex-col items-center justify-center min-h-[70vh] gap-8 animate-fade-in">
      {/* Animated waveform */}
      <div className="flex items-end gap-[3px] h-16">
        {bars.map((_, i) => (
          <div
            key={i}
            className="w-[3px] bg-accent rounded-full"
            style={{
              animation: `waveform ${0.8 + (i % 5) * 0.15}s ease-in-out ${(i * 0.08) % 0.6}s infinite`,
              height: `${(0.3 + (i % 7) * 0.1) * 100}%`,
            }}
          />
        ))}
      </div>

      <div className="text-center">
        <p className="font-mono text-xs text-muted mb-2">{fileName}</p>
        <p
          className="font-display text-xl text-white transition-all duration-500"
          style={{ fontWeight: 600 }}
        >
          {STEPS[stepIdx]}
        </p>
        <div className="flex justify-center gap-1.5 mt-4">
          {STEPS.map((_, i) => (
            <div
              key={i}
              className={`h-1 rounded-full transition-all duration-500 ${
                i <= stepIdx ? 'bg-accent w-6' : 'bg-subtle w-2'
              }`}
            />
          ))}
        </div>
      </div>

      <p className="font-mono text-xs text-muted/50 max-w-xs text-center">
        LLM evaluation may take 20–60s depending on conversation length
      </p>
    </div>
  )
}
