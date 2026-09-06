import { useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'

const NAV_ITEMS = [
  { to: '/', label: 'Data Overview', icon: '▦', end: true },
  { to: '/prediction', label: 'Risk Prediction', icon: '◉' },
  { to: '/explainability', label: 'Explainability', icon: '◈' },
  { to: '/rules', label: 'Business Rules', icon: '≡' },
  { to: '/chat', label: 'Ask the Data', icon: '▶' },
]

export default function Layout() {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <div className="flex min-h-screen bg-(--color-canvas)">
      {/* Mobile top bar */}
      <div className="fixed inset-x-0 top-0 z-20 flex items-center justify-between border-b border-(--color-border) bg-(--color-surface) px-4 py-3 md:hidden">
        <span className="text-sm font-semibold tracking-tight">Credit Risk Intelligence</span>
        <button
          onClick={() => setMobileOpen(!mobileOpen)}
          className="rounded-md border border-(--color-border) px-2.5 py-1.5 text-sm"
          aria-label="Toggle navigation"
        >
          {mobileOpen ? 'Close' : 'Menu'}
        </button>
      </div>

      {/* Mobile backdrop */}
      {mobileOpen && (
        <div
          onClick={() => setMobileOpen(false)}
          className="fixed inset-0 z-[5] bg-black/30 md:hidden"
          aria-hidden="true"
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-10 w-64 shrink-0 border-r border-(--color-border) bg-(--color-surface) pt-4 transition-transform duration-200 md:translate-x-0 ${
          mobileOpen ? 'translate-x-0' : '-translate-x-full'
        } md:pt-0`}
      >
        <div className="hidden px-6 py-6 md:block">
          <div className="text-base font-semibold tracking-tight text-(--color-ink)">
            Credit Risk Intelligence
          </div>
          <div className="mt-1 text-xs text-(--color-ink-faint)">Home Credit Default Risk</div>
        </div>
        <nav className="mt-14 flex flex-col gap-0.5 px-3 md:mt-2">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              onClick={() => setMobileOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors ${
                  isActive
                    ? 'bg-(--color-canvas) font-medium text-(--color-ink)'
                    : 'text-(--color-ink-muted) hover:bg-(--color-canvas) hover:text-(--color-ink)'
                }`
              }
            >
              <span className="text-base leading-none opacity-70">{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>

      {/* Main content */}
      <main className="min-w-0 flex-1 px-4 pt-20 pb-12 md:ml-64 md:px-10 md:pt-10">
        <div className="mx-auto max-w-5xl">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
