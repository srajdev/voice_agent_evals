import { useEffect, useRef } from 'react'
import type { MetricLabel } from '../types'

interface Props {
  score: number // 0–1
  label: MetricLabel
  size?: number
}

function scoreColor(score: number): string {
  if (score >= 0.85) return '#00d4aa'   // teal — excellent
  if (score >= 0.70) return '#4f8bff'   // blue — good
  if (score >= 0.50) return '#f5a623'   // amber — fair
  return '#ff4f6a'                       // red — poor
}

export default function ScoreGauge({ score, label, size = 180 }: Props) {
  const canvasRef = useRef<SVGCircleElement>(null)
  const color = scoreColor(score)
  const radius = (size / 2) * 0.72
  const strokeWidth = size * 0.075
  const circumference = 2 * Math.PI * radius
  const cx = size / 2
  const cy = size / 2

  // Animate on mount
  const dashOffset = circumference * (1 - score)

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="-rotate-90">
          {/* Track */}
          <circle
            cx={cx}
            cy={cy}
            r={radius}
            fill="none"
            stroke="rgba(255,255,255,0.05)"
            strokeWidth={strokeWidth}
          />
          {/* Glow duplicate */}
          <circle
            cx={cx}
            cy={cy}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth * 1.5}
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            strokeLinecap="round"
            opacity={0.15}
            style={{ transition: 'stroke-dashoffset 1.2s cubic-bezier(0.16, 1, 0.3, 1)' }}
          />
          {/* Main arc */}
          <circle
            ref={canvasRef}
            cx={cx}
            cy={cy}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            strokeLinecap="round"
            style={{ transition: 'stroke-dashoffset 1.2s cubic-bezier(0.16, 1, 0.3, 1)' }}
          />
          {/* Tick marks */}
          {[0, 0.25, 0.5, 0.75].map((pos) => {
            const angle = pos * 2 * Math.PI - Math.PI / 2
            const x1 = cx + (radius - strokeWidth) * Math.cos(angle)
            const y1 = cy + (radius - strokeWidth) * Math.sin(angle)
            const x2 = cx + (radius - strokeWidth * 0.4) * Math.cos(angle)
            const y2 = cy + (radius - strokeWidth * 0.4) * Math.sin(angle)
            return (
              <line
                key={pos}
                x1={x1} y1={y1} x2={x2} y2={y2}
                stroke="rgba(255,255,255,0.15)"
                strokeWidth="1"
              />
            )
          })}
        </svg>

        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span
            className="font-display leading-none"
            style={{
              fontSize: size * 0.22,
              fontWeight: 800,
              color,
              textShadow: `0 0 30px ${color}40`,
            }}
          >
            {Math.round(score * 100)}
          </span>
          <span className="font-mono text-muted" style={{ fontSize: size * 0.07 }}>
            / 100
          </span>
        </div>
      </div>

      <div className="text-center">
        <span
          className="inline-block px-3 py-0.5 rounded-full text-xs font-mono uppercase tracking-widest"
          style={{
            background: `${color}18`,
            color,
            border: `1px solid ${color}30`,
          }}
        >
          {label}
        </span>
      </div>
    </div>
  )
}
