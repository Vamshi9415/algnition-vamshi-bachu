import React from 'react'

function Card({ label, value, sub }) {
  return (
    <div className="bg-white dark:bg-ink-900 border border-ink-200 dark:border-ink-700 rounded-2xl p-5">
      <p className="text-ink-500 dark:text-ink-400 text-sm mb-1">{label}</p>
      <p className="text-3xl font-bold text-ink-950 dark:text-white">{value}</p>
      {sub && <p className="text-ink-400 dark:text-ink-500 text-xs mt-1">{sub}</p>}
    </div>
  )
}

export default function SummaryCards({ summary }) {
  const fmt = v => `$${Number(v).toLocaleString('en-US', { maximumFractionDigits: 0 })}`
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <Card label="Expected Revenue (P50)" value={fmt(summary.total_revenue_p50)} />
      <Card label="Conservative (P10)" value={fmt(summary.total_revenue_p10)} sub="10th percentile" />
      <Card label="Optimistic (P90)" value={fmt(summary.total_revenue_p90)} sub="90th percentile" />
      <Card label="Campaigns Forecasted" value={summary.campaigns_forecasted} sub={summary.channels?.join(', ')} />
    </div>
  )
}
