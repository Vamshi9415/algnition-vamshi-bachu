import React from 'react'

function Card({ label, value, sub }) {
  return (
    <div className="bg-slate-800 border border-slate-600 rounded-2xl p-5">
      <p className="text-slate-400 text-sm mb-1">{label}</p>
      <p className="text-3xl font-bold text-blue-400">{value}</p>
      {sub && <p className="text-slate-500 text-xs mt-1">{sub}</p>}
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
