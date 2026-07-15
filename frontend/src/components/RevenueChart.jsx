import React from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts'

export default function RevenueChart({ forecast }) {
  // Aggregate by date across all campaigns
  const byDate = {}
  forecast.forEach(row => {
    if (!byDate[row.date]) byDate[row.date] = { date: row.date, p10: 0, p50: 0, p90: 0 }
    byDate[row.date].p10 += row.revenue_p10
    byDate[row.date].p50 += row.revenue_p50
    byDate[row.date].p90 += row.revenue_p90
  })
  const data = Object.values(byDate).sort((a, b) => a.date.localeCompare(b.date))
  const fmt = v => `$${Number(v).toFixed(0)}`

  return (
    <div className="bg-slate-800 rounded-2xl p-6 border border-slate-600">
      <h2 className="text-xl font-semibold mb-4 text-slate-200">📈 Revenue Forecast (P10 / P50 / P90)</h2>
      <ResponsiveContainer width="100%" height={320}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id="gradP90" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 11 }} tickFormatter={d => d.slice(5)} />
          <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} tickFormatter={fmt} />
          <Tooltip formatter={(v) => fmt(v)} contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
          <Legend />
          <Area type="monotone" dataKey="p90" stroke="#3b82f6" fill="url(#gradP90)" strokeDasharray="4 2" name="P90 (Optimistic)" />
          <Area type="monotone" dataKey="p50" stroke="#60a5fa" fill="none" strokeWidth={2} name="P50 (Expected)" />
          <Area type="monotone" dataKey="p10" stroke="#94a3b8" fill="none" strokeDasharray="4 2" name="P10 (Conservative)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
