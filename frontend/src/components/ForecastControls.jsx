import React, { useEffect, useState } from 'react'
import axios from 'axios'

function addDays(isoDate, days) {
  const d = new Date(isoDate)
  d.setDate(d.getDate() + days)
  return d.toISOString().slice(0, 10)
}

export default function ForecastControls({ onResult, onLoading, onError }) {
  const [model, setModel] = useState(null)
  const [modelError, setModelError] = useState(null)
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [forecasting, setForecasting] = useState(false)

  useEffect(() => {
    axios.get('/api/v1/model')
      .then(res => {
        setModel(res.data)
        const start = res.data.earliest_forecastable_date
        setStartDate(start)
        setEndDate(addDays(start, 59))
      })
      .catch(e => setModelError(e.response?.data?.detail || e.message))
  }, [])

  const handleForecast = async () => {
    if (!startDate || !endDate) return
    setForecasting(true)
    onLoading(true)
    onError(null)
    try {
      const forecastRes = await axios.post(
        `/api/v1/forecast?start_date=${startDate}&end_date=${endDate}`
      )
      const { summary, forecast } = forecastRes.data

      const insightsRes = await axios.post('/api/v1/insights', { forecast, summary })

      onResult({
        status: 'success',
        summary,
        forecast,
        ai_insights: insightsRes.data,
      })
    } catch (e) {
      onError(e.response?.data?.detail || e.message)
    } finally {
      setForecasting(false)
      onLoading(false)
    }
  }

  if (modelError) {
    return (
      <div className="max-w-2xl mx-auto bg-ink-50 dark:bg-ink-900 border border-ink-900 dark:border-white rounded-lg p-4 text-ink-950 dark:text-white text-sm">
        Model unavailable: {modelError}
      </div>
    )
  }

  return (
    <div className="bg-white dark:bg-ink-900 rounded-2xl p-6 max-w-2xl mx-auto border border-ink-200 dark:border-ink-700">
      <h2 className="text-xl font-semibold mb-1 text-ink-950 dark:text-white">Forecast Range</h2>
      {model ? (
        <p className="text-xs text-ink-400 dark:text-ink-500 mb-4">
          {model.algorithm} &middot; {model.campaigns} campaigns &middot; {model.channels.join(', ')} &middot; trained {model.trained} &middot; WMAPE {model.wmape}% &middot; data through {model.data_through}
        </p>
      ) : (
        <p className="text-xs text-ink-400 dark:text-ink-500 mb-4 animate-pulse">Loading model info...</p>
      )}

      <div className="flex items-center gap-3 mb-4 flex-wrap">
        <label className="text-ink-700 dark:text-ink-300 text-sm">
          Start date
          <input
            type="date"
            value={startDate}
            min={model?.earliest_forecastable_date}
            onChange={e => setStartDate(e.target.value)}
            className="block w-40 bg-white dark:bg-ink-800 border border-ink-300 dark:border-ink-600 rounded-lg px-3 py-1 mt-1 text-ink-950 dark:text-ink-100"
          />
        </label>
        <label className="text-ink-700 dark:text-ink-300 text-sm">
          End date
          <input
            type="date"
            value={endDate}
            min={startDate || model?.earliest_forecastable_date}
            onChange={e => setEndDate(e.target.value)}
            className="block w-40 bg-white dark:bg-ink-800 border border-ink-300 dark:border-ink-600 rounded-lg px-3 py-1 mt-1 text-ink-950 dark:text-ink-100"
          />
        </label>
      </div>

      <button
        onClick={handleForecast}
        disabled={!startDate || !endDate || forecasting || !model}
        className="w-full bg-ink-950 hover:bg-ink-700 disabled:opacity-30 dark:bg-white dark:text-ink-950 dark:hover:bg-ink-200 text-white font-semibold py-3 rounded-xl transition"
      >
        {forecasting ? 'Forecasting...' : 'Generate Forecast'}
      </button>
    </div>
  )
}
