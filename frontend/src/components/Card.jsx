export default function Card({ children, className = '' }) {
  return (
    <div
      className={`rounded-xl border border-(--color-border) bg-(--color-surface) p-5 ${className}`}
    >
      {children}
    </div>
  )
}
