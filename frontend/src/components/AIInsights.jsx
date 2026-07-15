import React, { useState } from 'react'

export default function AIInsights({ insights }) {
  const [tab, setTab] = useState('summary')

  return (
    <div className="bg-slate-800 rounded-2xl p-6 border border-slate-600">
      <h2 className="text-xl font-semibold mb-4 text-slate-200">🤖 AI Insights</h2>
      <div className="flex gap-3 mb-4">
        <button
          onClick={() => setTab('summary')}
          className={`px-4 py-1.5 rounded-lg text-sm font-medium transition ${
            tab === 'summary' ? 'bg-blue-600 text-white' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
          }`}
        >
          Executive Summary
        </button>
        <button
          onClick={() => setTab('risks')}
          className={`px-4 py-1.5 rounded-lg text-sm font-medium transition ${
            tab === 'risks' ? 'bg-red-600 text-white' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
          }`}
        >
          Risk Analysis
        </button>
      </div>
      <div className="bg-slate-900 rounded-xl p-4 text-slate-300 text-sm leading-relaxed whitespace-pre-wrap">
        {tab === 'summary' ? insights?.executive_summary : insights?.risk_analysis}
      </div>
    </div>
  )
}
