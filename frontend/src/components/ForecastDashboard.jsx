import React from 'react'
import RevenueChart from './RevenueChart'
import SummaryCards from './SummaryCards'
import AIInsights from './AIInsights'
import CampaignTable from './CampaignTable'

export default function ForecastDashboard({ result }) {
  if (!result || result.status !== 'success') {
    return <div className="mt-8 text-center text-ink-600 dark:text-ink-300 font-medium">Forecast failed or invalid response.</div>
  }

  return (
    <div className="mt-10 space-y-8">
      <SummaryCards summary={result.summary} />
      <RevenueChart forecast={result.forecast} />
      <AIInsights insights={result.ai_insights} />
      <CampaignTable forecast={result.forecast} />
    </div>
  )
}
