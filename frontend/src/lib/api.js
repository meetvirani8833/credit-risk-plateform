// Base URL of the FastAPI backend. Set VITE_API_URL in Vercel's project
// settings to override for production; falls back to the deployed Render
// URL so the app works out of the box even before that env var is set.
const API_URL = import.meta.env.VITE_API_URL || 'https://credit-risk-api-l0nw.onrender.com'

async function request(path, options = {}) {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`${res.status} ${res.statusText}: ${text}`)
  }
  return res.json()
}

export const api = {
  edaInsights: () => request('/eda/insights'),
  chartUrl: (filename) => `${API_URL}/static/eda_figures/${filename}`,
  sampleApplicants: () => request('/predict/samples'),
  predict: (skIdCurr) =>
    request('/predict', { method: 'POST', body: JSON.stringify({ sk_id_curr: skIdCurr }) }),
  rules: () => request('/rules'),
  chat: (sessionId, message) =>
    request('/chat', { method: 'POST', body: JSON.stringify({ session_id: sessionId, message }) }),
}
