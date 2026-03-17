export type Speaker = 'user' | 'agent' | 'unknown'

export interface TimingInfo {
  speech_start_ms?: number
  speech_end_ms?: number
  ttfw_ms?: number
  source: string
}

export interface Turn {
  turn_id: string
  speaker: Speaker
  transcript?: string
  transcript_confidence?: number
  timing?: TimingInfo
  expected_intent?: string
  platform_metadata: Record<string, unknown>
}

export interface AudioInfo {
  original_file?: string
  duration_ms?: number
  sample_rate?: number
  channels?: number
  format?: string
  user_channel?: number
  agent_channel?: number
}

export interface PlatformInfo {
  platform: string
  call_id?: string
  agent_id?: string
  phone_number_from?: string
  phone_number_to?: string
}

export interface ScenarioConfig {
  scenario_id?: string
  description?: string
  expected_task?: string
  expected_intents: string[]
  completion_criteria?: string
  user_persona?: string
}

export interface VoiceTrace {
  trace_id: string
  created_at: string
  turns: Turn[]
  audio_info?: AudioInfo
  platform_info: PlatformInfo
  scenario?: ScenarioConfig
  call_start_at?: string
  call_end_at?: string
  metadata: Record<string, unknown>
}

export type MetricLabel =
  | 'excellent'
  | 'good'
  | 'fair'
  | 'poor'
  | 'completed'
  | 'partial'
  | 'failed'
  | 'not_applicable'
  | 'no_data'
  | 'error'
  | 'unknown'

export interface MetricScore {
  score: number
  label: MetricLabel
  reasoning: string
  details: Record<string, unknown>
}

export interface MetricResult {
  metric_name: string
  trace_id: string
  score: MetricScore
  raw_response?: string
}

export interface MetricSummaryEntry {
  score: number
  label: MetricLabel
}

export interface ReportSummary {
  overall_score: number
  overall_label: MetricLabel
  n_turns: number
  n_user_turns: number
  n_agent_turns: number
  duration_ms?: number
  platform: string
  scenario?: string
  metrics: Record<string, MetricSummaryEntry>
}

export interface EvaluationReport {
  report_id: string
  trace_id: string
  evaluated_at: string
  duration_ms: number
  results: MetricResult[]
  summary: ReportSummary
  trace?: VoiceTrace
}

export interface ReportListItem {
  report_id: string
  trace_id: string
  evaluated_at: string
  overall_score: number
  overall_label: MetricLabel
}

export interface ReportListResponse {
  total: number
  limit: number
  offset: number
  reports: ReportListItem[]
}

export type AppView = 'upload' | 'report' | 'history'
