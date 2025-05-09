import React, { useEffect, useState } from 'react'
import axios from 'axios'
import { Line } from 'react-chartjs-2'
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, TimeScale } from 'chart.js'
import ChartDataLabels from 'chartjs-plugin-datalabels';
import 'chartjs-adapter-date-fns'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend)
ChartJS.register(ChartDataLabels);

interface TrendDataItem {
  일자: string
  '수량(박스)': number
  '판매금액': number
}

interface BacktestDataItem {
  ds: string
  y: number
  yhat: number
  yhat_lower: number
  yhat_upper: number
  error_rate: number
}

interface ForecastResultItem {
  ds: string;
  yhat: number;
  yhat_lower: number;
  yhat_upper: number;
  yhat_corrected?: number;
}

interface TrendChartProps {
  item?: string
  category?: string
  fromDate?: string
  toDate?: string
}

const TrendChart: React.FC<TrendChartProps> = ({ item, category, fromDate, toDate }) => {
  const [data, setData] = useState<TrendDataItem[]>([])
  const [prevData, setPrevData] = useState<TrendDataItem[]>([])
  const [showPrevYear, setShowPrevYear] = useState<boolean>(false)
  const [showActualValues, setShowActualValues] = useState<boolean>(true)
  const [showForecast, setShowForecast] = useState<boolean>(false)
  const [forecastDays, setForecastDays] = useState<number>(3)
  const [showErrorRate, setShowErrorRate] = useState<boolean>(false)
  const [showSales, setShowSales] = useState<boolean>(false)
  const [actualMeasure, setActualMeasure] = useState<'quantity' | 'sales'>('quantity')
  const [prevMeasure, setPrevMeasure] = useState<'quantity' | 'sales'>('quantity')
  const [backtestData, setBacktestData] = useState<BacktestDataItem[]>([])
  const [futureForecast, setFutureForecast] = useState<ForecastResultItem[]>([])

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

  useEffect(() => {
    axios.post('/api/backtest', { item, category, from_date: fromDate, to_date: toDate })
      .then(res => setBacktestData(res.data))
      .catch(err => console.error(err))
  }, [item, category, fromDate, toDate])

  useEffect(() => {
    // Clear or fetch forecast based on last actual data
    if (!showForecast || !data.length) {
      setFutureForecast([])
      return
    }
    // Compute last actual data date
    const dates = data.map(d => d.일자.split('T')[0]).sort()
    const lastDate = dates[dates.length - 1]
    // Request future forecasts starting the day after last actual
    axios.post('/api/forecast', { item, category, periods: forecastDays, last_date: lastDate })
      .then(res => {
        const arr = Array.isArray(res.data) ? res.data : res.data.forecast
        setFutureForecast(arr as ForecastResultItem[])
      })
      .catch(err => console.error(err))
  }, [item, category, forecastDays, data, showForecast])

  // Determine the last actual data date for forecast basis
  const actualDates = data.map(d => d.일자.split('T')[0]).sort()
  const lastActual = actualDates.length ? actualDates[actualDates.length - 1] : undefined

  // Build continuous date labels from fromDate to toDate
  const dateLabels: string[] = []
  if (fromDate && toDate) {
    const start = new Date(fromDate)
    // 실제 데이터의 마지막 날짜와 toDate 중 더 이른 날짜까지만 x축 생성
    const lastDataDate = lastActual ? new Date(lastActual) : new Date(toDate)
    const end = lastDataDate < new Date(toDate) ? lastDataDate : new Date(toDate)
    for (let d = new Date(start); d <= end; d.setDate(d.getDate() + 1)) {
      dateLabels.push(d.toISOString().split('T')[0])
    }
  }
  // Build future forecast dates starting the day after last actual data
  const futureDates: string[] = []
  if (lastActual && showForecast) {
    const start = new Date(lastActual)
    for (let i = 1; i <= forecastDays; i++) {
      const d = new Date(start)
      d.setDate(d.getDate() + i)
      futureDates.push(d.toISOString().split('T')[0])
    }
  }
  const allDates = [...dateLabels, ...futureDates]
  const labels = allDates.map(d => d.slice(5))

  // Map historical and future data to arrays
  const dataMap = new Map(data.map(d => [d.일자.split('T')[0], d['수량(박스)']]))
  const salesMap = new Map(data.map(d => [d.일자.split('T')[0], d['판매금액']]))
  const errorMap = new Map(backtestData.map(b => [b.ds.split('T')[0], b.error_rate]))
  // Historical values arrays
  const histValues = dateLabels.map(d => dataMap.get(d) ?? null)
  const histSales = dateLabels.map(d => salesMap.get(d) ?? null)
  const histError = dateLabels.map(d => errorMap.get(d) ?? null)
  // Forecast mapping
  const fcMap = new Map(futureForecast.map(f => [f.ds.split('T')[0], (f.yhat_corrected ?? f.yhat)]))
  const fcValues = futureDates.map(d => fcMap.get(d) ?? null)

  // Compute linear regression for trend line on historical values (exclude nulls)
  const validPoints = histValues
    .map((v, i) => ({ v, i }))
    .filter(pt => pt.v != null) as { v: number; i: number }[]
  const xArr = validPoints.map(pt => pt.i)
  const yArr = validPoints.map(pt => pt.v)
  const n = xArr.length
  const xSum = xArr.reduce((acc, x) => acc + x, 0)
  const ySum = yArr.reduce((acc, y) => acc + y, 0)
  const xySum = xArr.reduce((acc, x, idx) => acc + x * yArr[idx], 0)
  const xxSum = xArr.reduce((acc, x) => acc + x * x, 0)
  const slope = n ? (n * xySum - xSum * ySum) / (n * xxSum - xSum * xSum) : 0
  const intercept = n ? (ySum - slope * xSum) / n : 0

  // Prev-year mapping and padded arrays
  const prevMapQuantity = new Map(prevData.map(d => [d.일자.split('T')[0], d['수량(박스)']]))
  const prevMapSales = new Map(prevData.map(d => [d.일자.split('T')[0], d['판매금액']]))
  // Helper: get previous year date string for a given date string
  const prevYearDate = (d: string) => {
    const dt = new Date(d);
    dt.setFullYear(dt.getFullYear() - 1);
    return dt.toISOString().split('T')[0];
  };
  const paddedPrev = [...dateLabels.map(d => prevMapQuantity.get(prevYearDate(d)) ?? null), ...futureDates.map(() => null)]
  const paddedPrevSales = [...dateLabels.map(d => prevMapSales.get(prevYearDate(d)) ?? null), ...futureDates.map(() => null)]
  // Combined padded arrays
  const paddedValues = [...histValues, ...futureDates.map(() => null)]
  const paddedSales = [...histSales, ...futureDates.map(() => null)]
  const paddedError = [...histError, ...futureDates.map(() => null)]
  const paddedForecast = [...dateLabels.map(() => null), ...fcValues]
  // Full trendline values (historical + nulls)
  const trendlineFull = [...histValues.map((_, idx) => intercept + slope * idx), ...futureDates.map(() => null)]

  // Build datasets in control order: Actual -> Prev-year -> Forecast -> Error -> Sales -> Trendline
  const datasets: any[] = []
  // Actual series
  if (showActualValues) {
    datasets.push({
      label: actualMeasure === 'quantity' ? '수량(박스)' : '판매금액',
      data: paddedValues,
      yAxisID: actualMeasure === 'quantity' ? 'quantity' : 'sales',
      borderColor: actualMeasure === 'quantity' ? 'rgba(75,192,192,1)' : 'rgba(255,99,132,1)',
      fill: false,
      tension: 0.4
    })
  }
  // Prev-year series
  if (showPrevYear) {
    datasets.push({
      label: prevMeasure === 'quantity' ? '전년동기 수량' : '전년동기 판매금액',
      data: prevMeasure === 'quantity' ? paddedPrev : paddedPrevSales,
      yAxisID: prevMeasure === 'quantity' ? 'quantity' : 'sales',
      borderColor: 'rgba(153,102,255,1)',
      fill: false,
      tension: 0.4
    })
  }
  // Forecast series
  if (showForecast) {
    datasets.push({
      label: `예측값(${forecastDays}일)`,
      data: paddedForecast,
      yAxisID: 'quantity',
      borderColor: 'rgba(255,159,64,1)',
      fill: false,
      tension: 0.4
    })
  }
  // Error rate series
  if (showErrorRate) {
    datasets.push({
      label: '오차율 (%)',
      data: paddedError,
      yAxisID: 'error',
      borderColor: 'rgba(255,206,86,1)',
      fill: false,
      tension: 0.4
    })
  }
  // Sales series
  if (showSales) {
    datasets.push({
      label: '판매금액',
      data: paddedSales,
      yAxisID: 'sales',
      borderColor: 'rgba(255,99,132,1)',
      fill: false,
      tension: 0.4
    })
  }
  // Trendline
  if (showActualValues || showPrevYear || showForecast || showErrorRate || showSales) {
    datasets.push({
      label: '추세선',
      data: trendlineFull,
      yAxisID: 'quantity',
      borderColor: 'rgba(0,0,0,0.6)',
      borderDash: [5,5],
      fill: false,
      pointRadius: 0,
      tension: 0
    })
  }
  const chartData = { labels, datasets }

  return (
    <div className="chart-container">
      {/* Chart series controls */}
      <fieldset className="chart-controls">
        <legend>데이터 시리즈</legend>
        <label>실제값:
          <select value={actualMeasure} onChange={e => setActualMeasure(e.target.value as 'quantity'|'sales')}>
            <option value="quantity">수량(박스)</option>
            <option value="sales">판매금액</option>
          </select>
        </label>
        <label>전년동기:
          <select value={prevMeasure} onChange={e => setPrevMeasure(e.target.value as 'quantity'|'sales')}>
            <option value="quantity">수량(박스)</option>
            <option value="sales">판매금액</option>
          </select>
        </label>
        <label>전년동기 표시:
          <input type="checkbox" checked={showPrevYear} onChange={e => setShowPrevYear(e.target.checked)} />
        </label>
        <label>예측값 표시:
          <input type="checkbox" checked={showForecast} onChange={e => setShowForecast(e.target.checked)} />
          일수:
          <select value={forecastDays} onChange={e => setForecastDays(Number(e.target.value))}>
            <option value={3}>3일</option>
            <option value={5}>5일</option>
            <option value={7}>7일</option>
          </select>
        </label>
        <label>오차율 표시:
          <input type="checkbox" checked={showErrorRate} onChange={e => setShowErrorRate(e.target.checked)} />
        </label>
        <label>판매금액 표시:
          <input type="checkbox" checked={showSales} onChange={e => setShowSales(e.target.checked)} />
        </label>
      </fieldset>
      <h2>일별 출고 수량 및 판매금액 추이</h2>
      <Line
        data={chartData}
        options={{
          responsive: true,
          scales: {
            quantity: { type: 'linear', position: 'left', title: { display: true, text: '수량(박스)' } },
            sales: {
              type: 'linear',
              position: 'right',
              display: showSales,
              title: { display: true, text: '판매금액(백만)' },
              grid: { drawOnChartArea: false },
              ticks: { callback: (val: any) => `${(val / 1000000).toFixed(1)}M` }
            },
            error: {
              type: 'linear',
              position: 'right',
              offset: true,
              display: showErrorRate,
              title: { display: true, text: '오차율 (%)' },
              grid: { drawOnChartArea: false },
              ticks: { callback: (val: any) => `${val.toFixed(1)}%` }
            }
          },
          plugins: {
            legend: { labels: { filter: (item: any, chart: any) => {
                // maintain order: actual, prev-year, forecast, error, sales, trendline
                return true;
            } } },
            datalabels: {
              display: (context: any) => {
                if (context.dataset.label === '추세선') {
                  const idx = context.dataIndex;
                  return idx === 0 || idx === dateLabels.length - 1;
                }
                // 모든 시리즈(추세선 제외)는 항상 표시
                return context.dataset.label !== undefined;
              },
              align: 'end' as const,
              formatter: (value: number, context: any) => {
                // 판매금액 시리즈는 백만원 단위로 1자리 소수
                if (context.dataset.label && context.dataset.label.includes('판매금액')) {
                  return (value / 1_000_000).toFixed(1) + 'M';
                }
                return Math.round(value);
              }
            },
            tooltip: { callbacks: { label: ctx => {
              if (ctx.dataset.label === '판매금액') return `${(ctx.parsed.y / 1000000).toFixed(2)}M`;
              if (ctx.dataset.label?.startsWith('오차율')) return `${ctx.parsed.y.toFixed(1)}%`;
              return `${ctx.dataset.label}: ${ctx.parsed.y}`;
            } } }
          }
        }}
      />
    </div>
  )
}

export default TrendChart