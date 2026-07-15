import React, { useState } from 'react'
import UploadPanel from './components/UploadPanel'
import ForecastDashboard from './components/ForecastDashboard'

export default function App() {
  const [forecastResult, setForecastResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  return (
    <div className="min-h-screen p-6">
      <header className="mb-8 text-center">
        <h1 className="text-4xl font-bold text-blue-400">⚡ AIgnition Forecast Studio</h1>
        <p className="text-slate-400 mt-2 text-lg">AI-Assisted Probabilistic Revenue Forecasting</p>
      </header>

      <UploadPanel
        onResult={setForecastResult}
        onLoading={setLoading}
        onError={setError}
      />

      {loading && (
        <div className="mt-10 text-center text-blue-300 text-xl animate-pulse">
          ⏳ Running forecast pipeline...
        </div>
      )}

      {error && (
        <div className="mt-6 bg-red-900/50 border border-red-500 rounded-lg p-4 text-red-300">
          ❌ {error}
        </div>
      )}

      {forecastResult && !loading && (
        <ForecastDashboard result={forecastResult} />
      )}
    </div>
  )
}
