import type { AppView } from '../types'

interface Props {
  currentView: AppView
  onNav: (view: AppView) => void
}

export default function NavBar({ currentView, onNav }: Props) {
  return (
    <header className="relative z-20 border-b border-border bg-surface/80 backdrop-blur-sm">
      <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
        {/* Logo */}
        <button
          onClick={() => onNav('upload')}
          className="flex items-center gap-3 group"
        >
          <WaveformIcon />
          <span
            className="font-display font-700 text-lg tracking-tight text-white group-hover:text-accent transition-colors"
            style={{ fontWeight: 700 }}
          >
            voice<span className="text-accent">evals</span>
          </span>
        </button>

        {/* Nav links */}
        <nav className="flex items-center gap-1">
          <NavLink active={currentView === 'upload'} onClick={() => onNav('upload')}>
            Evaluate
          </NavLink>
          <NavLink active={currentView === 'history'} onClick={() => onNav('history')}>
            History
          </NavLink>
        </nav>
      </div>
    </header>
  )
}

function NavLink({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all duration-150 ${
        active
          ? 'bg-accent/10 text-accent'
          : 'text-muted hover:text-white hover:bg-subtle'
      }`}
    >
      {children}
    </button>
  )
}

function WaveformIcon() {
  const bars = [0.4, 0.7, 1, 0.8, 0.5, 0.9, 0.6, 0.3, 0.7, 1, 0.5]
  return (
    <div className="flex items-center gap-[2px] h-5">
      {bars.map((h, i) => (
        <div
          key={i}
          className="w-[2px] bg-accent rounded-full"
          style={{ height: `${h * 100}%` }}
        />
      ))}
    </div>
  )
}
