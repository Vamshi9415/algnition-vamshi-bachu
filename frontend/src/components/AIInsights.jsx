import React, { useState } from 'react'

export default function AIInsights({ insights }) {
  const [tab, setTab] = useState('summary')

  return (
    <div className="bg-white dark:bg-ink-900 rounded-2xl p-6 border border-ink-200 dark:border-ink-700">
      <h2 className="text-xl font-semibold mb-4 text-ink-950 dark:text-white">AI Insights</h2>
      <div className="flex gap-3 mb-4">
        <button
          onClick={() => setTab('summary')}
          className={`px-4 py-1.5 rounded-lg text-sm font-medium transition ${
            tab === 'summary'
              ? 'bg-ink-950 text-white dark:bg-white dark:text-ink-950'
              : 'bg-ink-100 text-ink-600 hover:bg-ink-200 dark:bg-ink-800 dark:text-ink-300 dark:hover:bg-ink-700'
          }`}
        >
          Executive Summary
        </button>
        <button
          onClick={() => setTab('risks')}
          className={`px-4 py-1.5 rounded-lg text-sm font-medium transition ${
            tab === 'risks'
              ? 'bg-ink-950 text-white dark:bg-white dark:text-ink-950'
              : 'bg-ink-100 text-ink-600 hover:bg-ink-200 dark:bg-ink-800 dark:text-ink-300 dark:hover:bg-ink-700'
          }`}
        >
          Risk Analysis
        </button>
      </div>
      <div className="bg-ink-50 dark:bg-ink-950 border border-ink-200 dark:border-ink-800 rounded-xl p-4 text-ink-700 dark:text-ink-300 text-sm leading-relaxed whitespace-pre-wrap">
        {tab === 'summary' ? insights?.executive_summary : insights?.risk_analysis}
      </div>
    </div>
  )
}
