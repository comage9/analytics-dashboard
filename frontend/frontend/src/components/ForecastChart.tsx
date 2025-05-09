import React, { useEffect, useState } from 'react'
import axios from 'axios'
import { Line } from 'react-chartjs-2'
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js'
import ChartDataLabels from 'chartjs-plugin-datalabels'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend)
ChartJS.register(ChartDataLabels)

interface ForecastDataItem {
  ds: string
  yhat: number
  yhat_lower: number
  yhat_upper: number
  yhat_corrected?: number
}

interface ForecastChartProps {
  item?: string
  category?: string
  periods?: number
  fromDate?: string
  lastDate?: string
  useCustom?: boolean
}

const ForecastChart: React.FC<ForecastChartProps> = ({ item, category, periods = 30, fromDate, lastDate, useCustom = false }) => {
  const [data, setData] = useState<ForecastDataItem[]>([])

  useEffect(() => {
    axios.post('/api/forecast', { item, category, periods, from_date: fromDate, last_date: lastDate, use_custom: useCustom })
      .then(res => {
        const payload = Array.isArray(res.data) ? res.data : res.data.forecast;
        setData(payload || []);
      })
      .catch(err => console.error(err))
  }, [item, category, periods, fromDate, lastDate, useCustom])

  const labels = data.map(item => item.ds)
  const useCorrected = data.length > 0 && data[0].yhat_corrected !== undefined
  const values = data.map(item => Math.max(0, useCorrected ? (item.yhat_corrected as number) : item.yhat))
  const lower = data.map(item => Math.max(0, item.yhat_lower))
  const upper = data.map(item => Math.max(0, item.yhat_upper))

  const n = values.length
  const xSum = values.reduce((acc, _, idx) => acc + idx, 0)
  const ySum = values.reduce((acc, y) => acc + y, 0)
  const xySum = values.reduce((acc, y, idx) => acc + idx * y, 0)
  const xxSum = values.reduce((acc, _, idx) => acc + idx * idx, 0)
  const slope = (n * xySum - xSum * ySum) / (n * xxSum - xSum * xSum) || 0
  const intercept = n ? (ySum - slope * xSum) / n : 0
  const trendlinePoints = values.map((_, idx) => intercept + slope * idx)

  const chartData = {
    labels,
    datasets: [
      {
        label: useCorrected
          ? '보정된 예측 수량(박스)'
          : '예측 수량(박스)',
        data: values,
        borderColor: 'rgba(255,99,132,1)',
        backgroundColor: 'rgba(255,99,132,0.2)',
        fill: true,
      },
      {
        label: '하한선',
        data: lower,
        borderColor: 'rgba(54,162,235,0.4)',
        borderDash: [5, 5],
        fill: false,
      },
      {
        label: '상한선',
        data: upper,
        borderColor: 'rgba(54,162,235,0.4)',
        borderDash: [5, 5],
        fill: false,
      },
      {
        label: '추세선',
        data: trendlinePoints,
        borderColor: 'rgba(0,0,0,0.6)',
        borderDash: [5, 5],
        fill: false,
        pointRadius: 0,
        tension: 0
      }
    ]
  }

  return (
    <div className="chart-container">
      <h2>예측 수량(박스) (다음 {periods}일)</h2>
      <div className="forecast-info">
        <p>예측 기반: {useCustom
          ? '커스텀 Prophet (changepoint_prior_scale=0.5, seasonality_prior_scale=10.0, 월별·분기별 seasonalities)'
          : '기본 Prophet 모델'}
        </p>
      </div>
      <Line
        data={chartData}
        options={{
          plugins: {
            datalabels: {
              display: context => context.dataset.label === '추세선',
              align: 'end' as const,
              formatter: (value: number) => Math.round(value)
            }
          }
        }}
      />
    </div>
  )
}

export default ForecastChart