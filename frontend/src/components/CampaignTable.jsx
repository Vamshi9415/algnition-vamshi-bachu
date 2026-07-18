import React, { useState } from 'react'

export default function CampaignTable({ forecast }) {
  const [sortCol, setSortCol] = useState('revenue_p50')

  // Aggregate by campaign
  const bycamp = {}
  forecast.forEach(row => {
    const k = `${row.channel}||${row.campaign_name}`
    if (!bycamp[k]) bycamp[k] = { channel: row.channel, campaign_name: row.campaign_name, revenue_p10: 0, revenue_p50: 0, revenue_p90: 0, days: 0 }
    bycamp[k].revenue_p10 += row.revenue_p10
    bycamp[k].revenue_p50 += row.revenue_p50
    bycamp[k].revenue_p90 += row.revenue_p90
    bycamp[k].days += 1
  })

  const rows = Object.values(bycamp).sort((a, b) => b[sortCol] - a[sortCol])
  const fmt = v => `$${Number(v).toLocaleString('en-US', { maximumFractionDigits: 0 })}`

  return (
    <div className="bg-white dark:bg-ink-900 rounded-2xl p-6 border border-ink-200 dark:border-ink-700 overflow-x-auto">
      <h2 className="text-xl font-semibold mb-4 text-ink-950 dark:text-white">Campaign Forecast Breakdown</h2>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-ink-500 dark:text-ink-400 border-b border-ink-200 dark:border-ink-700">
            <th className="text-left py-2 pr-4">Channel</th>
            <th className="text-left py-2 pr-4">Campaign</th>
            <th className="text-right py-2 pr-4 cursor-pointer hover:text-ink-950 dark:hover:text-white" onClick={() => setSortCol('revenue_p10')}>P10</th>
            <th className="text-right py-2 pr-4 cursor-pointer hover:text-ink-950 dark:hover:text-white" onClick={() => setSortCol('revenue_p50')}>P50 ▼</th>
            <th className="text-right py-2 cursor-pointer hover:text-ink-950 dark:hover:text-white" onClick={() => setSortCol('revenue_p90')}>P90</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-ink-100 dark:border-ink-800 hover:bg-ink-50 dark:hover:bg-ink-800/60 transition">
              <td className="py-2 pr-4 text-ink-500 dark:text-ink-400 capitalize">{r.channel}</td>
              <td className="py-2 pr-4 text-ink-950 dark:text-ink-100">{r.campaign_name}</td>
              <td className="py-2 pr-4 text-right text-ink-500 dark:text-ink-400">{fmt(r.revenue_p10)}</td>
              <td className="py-2 pr-4 text-right text-ink-950 dark:text-white font-semibold">{fmt(r.revenue_p50)}</td>
              <td className="py-2 text-right text-ink-500 dark:text-ink-400">{fmt(r.revenue_p90)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
