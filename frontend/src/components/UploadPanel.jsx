import React, { useState } from 'react'
import axios from 'axios'

export default function UploadPanel({ onResult, onLoading, onError }) {
  const [files, setFiles] = useState([])
  const [horizon, setHorizon] = useState(60)

  const handleUpload = async () => {
    if (!files.length) return
    onLoading(true)
    onError(null)
    const formData = new FormData()
    files.forEach(f => formData.append('files', f))
    try {
      const res = await axios.post(
        `/api/v1/forecast?horizon_days=${horizon}`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      )
      onResult(res.data)
    } catch (e) {
      onError(e.response?.data?.detail || e.message)
    } finally {
      onLoading(false)
    }
  }

  return (
    <div className="bg-slate-800 rounded-2xl p-6 max-w-2xl mx-auto border border-slate-600">
      <h2 className="text-xl font-semibold mb-4 text-slate-200">📂 Upload Ad Data</h2>
      <input
        type="file"
        multiple
        accept=".csv"
        onChange={e => setFiles(Array.from(e.target.files))}
        className="block w-full text-sm text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-blue-600 file:text-white hover:file:bg-blue-500 mb-4"
      />
      {files.length > 0 && (
        <ul className="text-sm text-slate-400 mb-3 list-disc list-inside">
          {files.map(f => <li key={f.name}>{f.name}</li>)}
        </ul>
      )}
      <div className="flex items-center gap-3 mb-4">
        <label className="text-slate-300 text-sm">Forecast horizon (days):</label>
        <input
          type="number"
          value={horizon}
          min={7}
          max={180}
          onChange={e => setHorizon(Number(e.target.value))}
          className="w-20 bg-slate-700 border border-slate-500 rounded-lg px-3 py-1 text-slate-200"
        />
      </div>
      <button
        onClick={handleUpload}
        disabled={!files.length}
        className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white font-semibold py-3 rounded-xl transition"
      >
        🚀 Generate Forecast
      </button>
    </div>
  )
}
