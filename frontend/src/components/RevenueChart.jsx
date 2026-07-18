import React from 'react'
import Plot from 'react-plotly.js'

const isDark = typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches

const palette = isDark
  ? { p50: '#fafafa', p90: '#a3a3a3', p10: '#525252', band: 'rgba(163,163,163,0.12)', grid: '#262626', axis: '#a3a3a3', paper: '#171717', text: '#f5f5f5' }
  : { p50: '#171717', p90: '#737373', p10: '#a3a3a3', band: 'rgba(115,115,115,0.10)', grid: '#e5e5e5', axis: '#737373', paper: '#ffffff', text: '#171717' }

export default function RevenueChart({ forecast }) {
  const byDate = {}
  forecast.forEach(row => {
    if (!byDate[row.date]) byDate[row.date] = { date: row.date, p10: 0, p50: 0, p90: 0 }
    byDate[row.date].p10 += row.revenue_p10
    byDate[row.date].p50 += row.revenue_p50
    byDate[row.date].p90 += row.revenue_p90
  })
  const data = Object.values(byDate).sort((a, b) => a.date.localeCompare(b.date))
  const dates = data.map(d => d.date)

  const traces = [
    {
      x: dates,
      y: data.map(d => d.p90),
      name: 'P90 (Optimistic)',
      mode: 'lines',
      line: { color: palette.p90, width: 1.5, dash: 'dot' },
      hovertemplate: '%{x}<br>P90: $%{y:,.0f}<extra></extra>',
    },
    {
      x: dates,
      y: data.map(d => d.p10),
      name: 'P10 (Conservative)',
      mode: 'lines',
      line: { color: palette.p10, width: 1.5, dash: 'dot' },
      fill: 'tonexty',
      fillcolor: palette.band,
      hovertemplate: '%{x}<br>P10: $%{y:,.0f}<extra></extra>',
    },
    {
      x: dates,
      y: data.map(d => d.p50),
      name: 'P50 (Expected)',
      mode: 'lines',
      line: { color: palette.p50, width: 2.5 },
      hovertemplate: '%{x}<br>P50: $%{y:,.0f}<extra></extra>',
    },
  ]

  return (
    <div className="bg-white dark:bg-ink-900 rounded-2xl p-6 border border-ink-200 dark:border-ink-700">
      <h2 className="text-xl font-semibold mb-4 text-ink-950 dark:text-white">Revenue Forecast (P10 / P50 / P90)</h2>
      <Plot
        data={traces}
        layout={{
          autosize: true,
          height: 340,
          margin: { l: 60, r: 20, t: 10, b: 40 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          font: { color: palette.text, size: 12 },
          xaxis: { gridcolor: palette.grid, linecolor: palette.grid, tickfont: { color: palette.axis } },
          yaxis: { gridcolor: palette.grid, linecolor: palette.grid, tickfont: { color: palette.axis }, tickprefix: '$', separatethousands: true },
          legend: { orientation: 'h', y: -0.2, font: { color: palette.text } },
          hovermode: 'x unified',
        }}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%' }}
        useResizeHandler
      />
    </div>
  )
}
