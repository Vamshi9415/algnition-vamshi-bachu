import React, { useState } from 'react'
import ForecastControls from './components/ForecastControls'
import ForecastDashboard from './components/ForecastDashboard'

export default function App() {
  const [forecastResult, setForecastResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  return (
    <div className="min-h-screen bg-white dark:bg-ink-950 p-6">
      <header className="mb-8 text-center">
        <h1 className="text-4xl font-bold text-ink-950 dark:text-white tracking-tight">AIgnition Forecast Studio</h1>
        <p className="text-ink-500 dark:text-ink-400 mt-2 text-lg">AI-Assisted Probabilistic Revenue Forecasting</p>
      </header>

      <ForecastControls
        onResult={setForecastResult}
        onLoading={setLoading}
        onError={setError}
      />

      {loading && (
        <div className="mt-10 text-center text-ink-600 dark:text-ink-300 text-xl animate-pulse">
          Running forecast pipeline...
        </div>
      )}

      {error && (
        <div className="mt-6 max-w-2xl mx-auto bg-ink-50 dark:bg-ink-900 border border-ink-900 dark:border-white rounded-lg p-4 text-ink-950 dark:text-white">
          {error}
        </div>
      )}

      {forecastResult && !loading && (
        <ForecastDashboard result={forecastResult} />
      )}
    </div>
  )
}
