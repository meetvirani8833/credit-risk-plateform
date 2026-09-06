import { useState } from 'react'

// Charts (especially the multi-panel ones) are illegible at thumbnail size.
// Shown large and full-width by default, click to view at native resolution
// in a lightbox instead of squeezing dense multi-panel plots into a fixed
// small box.
export default function ChartImage({ src, alt }) {
  const [zoomed, setZoomed] = useState(false)

  return (
    <>
      <button
        onClick={() => setZoomed(true)}
        className="block w-full cursor-zoom-in rounded-lg border border-(--color-border)"
      >
        <img src={src} alt={alt} className="w-full rounded-lg" />
      </button>

      {zoomed && (
        <div
          onClick={() => setZoomed(false)}
          className="fixed inset-0 z-50 flex cursor-zoom-out items-center justify-center bg-black/70 p-6"
        >
          <img
            src={src}
            alt={alt}
            className="max-h-full max-w-full rounded-lg bg-white p-2 shadow-2xl"
          />
        </div>
      )}
    </>
  )
}
