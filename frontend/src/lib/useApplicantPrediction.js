import { useEffect, useState } from 'react'
import { api } from './api'

// Shared between the Prediction and Explainability pages, both let the
// user pick from the same sample applicants and score them, they just
// present the result differently.
export function useApplicantPrediction() {
  const [samples, setSamples] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    api
      .sampleApplicants()
      .then((list) => {
        setSamples(list)
        if (list.length) setSelectedId(list[0].sk_id_curr)
      })
      .catch((e) => setError(e.message))
  }, [])

  useEffect(() => {
    if (selectedId == null) return
    setLoading(true)
    setError(null)
    api
      .predict(selectedId)
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [selectedId])

  return { samples, selectedId, setSelectedId, result, loading, error }
}
