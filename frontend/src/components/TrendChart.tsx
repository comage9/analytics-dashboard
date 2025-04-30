import React, { useEffect, useState } from 'react'
import axios from 'axios'
import { Line } from 'react-chartjs-2'
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js'
import ChartDataLabels from 'chartjs-plugin-datalabels';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend)
ChartJS.register(ChartDataLabels);

interface TrendDataItem {
  일자: string
  '수량(박스)': number
  '판매금액': number
}

interface TrendChartProps {
  item?: string
  category?: string
  fromDate?: string
  toDate?: string
  showPrevTrend?: boolean
}

const TrendChart: React.FC<TrendChartProps> = ({ item, category, fromDate, toDate, showPrevTrend }) => {
  const [data, setData] = useState<TrendDataItem[]>([])
  const [prevData, setPrevData] = useState<TrendDataItem[]>([])

  useEffect(() => {
    axios.post('/api/trend', { item, category, from_date: fromDate, to_date: toDate })
      .then(res => setData(res.data))
      .catch(err => console.error(err))
    // fetch previous-year same period
    if (fromDate && toDate) {
      const f = new Date(fromDate), t = new Date(toDate);
      const pf = new Date(f.setFullYear(f.getFullYear() - 1));
      const pt = new Date(t.setFullYear(t.getFullYear() - 1));
      const prevFrom = pf.toISOString().split('T')[0];
      const prevTo = pt.toISOString().split('T')[0];
      axios.post('/api/trend', { item, category, from_date: prevFrom, to_date: prevTo })
        .then(res => setPrevData(res.data))
        .catch(err => console.error(err));
    }
  }, [item, category, fromDate, toDate])

  const labels = data.map(d => {
    const day = d.일자.split('T')[0]
    return day.slice(5)
  })
  const values = data.map(d => d['수량(박스)'] || 0)
  const salesValues = data.map(d => d['판매금액'] || 0)

  // Compute linear regression for trend line
  const n = values.length
  const xSum = values.reduce((acc, _, idx) => acc + idx, 0)
  const ySum = values.reduce((acc, y) => acc + y, 0)
  const xySum = values.reduce((acc, y, idx) => acc + idx * y, 0)
  const xxSum = values.reduce((acc, _, idx) => acc + idx * idx, 0)
  const slope = (n * xySum - xSum * ySum) / (n * xxSum - xSum * xSum) || 0
  const intercept = n ? (ySum - slope * xSum) / n : 0
  const trendlinePoints = values.map((_, idx) => intercept + slope * idx)

  // prepare previous-year trend data if available
  const prevTrendValues = prevData.map(d => d['수량(박스)'] || 0)
  // constant avg line if no trend toggle
  const prevArray = prevData.map(d => d['수량(박스)'] || 0)
  const prevAvg = prevArray.length > 0 ? prevArray.reduce((a, b) => a + b, 0) / prevArray.length : 0
  const prevValues = labels.map(() => prevAvg)

  const chartData = {
    labels,
    datasets: [
      // Conditional previous-year dataset: avg or trend
      {
        label: showPrevTrend ? '전년도 추세' : '전년 동기 평균',
        data: showPrevTrend && prevTrendValues.length === data.length ? prevTrendValues : prevValues,
        yAxisID: 'quantity',
        borderColor: 'rgba(153,102,255,1)',
        backgroundColor: 'rgba(153,102,255,0.2)',
        fill: false,
        tension: showPrevTrend ? 0.4 : 0,
        borderDash: showPrevTrend ? [] : [5,5]
      },
      { // Actual trend
        label: '수량(박스)',
        data: values,
        yAxisID: 'quantity',
        borderColor: 'rgba(75,192,192,1)',
        backgroundColor: 'rgba(75,192,192,0.2)',
        fill: false,
        tension: 0.4,
        datalabels: {
          // Quantity as integer with teal color
          display: true,
          color: 'rgba(75,192,192,1)',
          align: 'top' as const,
          formatter: (value: number) => Math.round(value)
        }
      },
      { // Sales
        label: '판매금액',
        data: salesValues,
        yAxisID: 'sales',
        borderColor: 'rgba(255,99,132,1)',
        backgroundColor: 'rgba(255,99,132,0.2)',
        fill: false,
        tension: 0.4,
        datalabels: {
          display: true,
          align: 'top' as const,
          formatter: (value: number) => `${Math.round(value / 1000000)}M`
        }
      },
      { // Regression trend line
        label: '추세선',
        data: trendlinePoints,
        yAxisID: 'quantity',
        borderColor: 'rgba(0,0,0,0.6)',
        borderDash: [5,5],
        fill: false,
        pointRadius: 0,
        tension: 0,
        datalabels: {
          // Trendline labels in dark gray
          display: true,
          color: 'rgba(80,80,80,1)',
          align: 'end' as const,
          formatter: (value: number) => Math.round(value)
        }
      }
    ]
  }

  return (
    <div className="chart-container">
      <h2>일별 출고 수량 및 판매금액 추이</h2>
      <Line
        data={chartData}
        options={{
          responsive: true,
          scales: {
            quantity: { type: 'linear', position: 'left', title: { display: true, text: '수량(박스)' } },
            sales: {
              type: 'linear', position: 'right',
              title: { display: true, text: '판매금액(백만)' },
              grid: { drawOnChartArea: false },
              ticks: {
                callback: (val: any) => `${Math.round(val / 1000000)}`
              }
            }
          },
          plugins: {
            datalabels: {
              // Regression labels only as integers
              display: (context: any) => context.dataset.label === '추세선',
              align: 'end' as const,
              formatter: (value: number) => Math.round(value),
            },
            tooltip: {
              callbacks: {
                label: ctx => {
                  if (ctx.dataset.label === '판매금액') return `${(ctx.parsed.y / 1000000).toFixed(2)}M`;
                  return `${ctx.dataset.label}: ${ctx.parsed.y}`;
                }
              }
            }
          }
        }}
      />
    </div>
  )
}

export default TrendChart