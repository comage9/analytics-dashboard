import React, { useState, useEffect } from 'react'
import axios from 'axios'
import TrendChart from './components/TrendChart'
import ForecastChart from './components/ForecastChart'
import ItemCategorySelector from './components/ItemCategorySelector'
import './App.css'

// Removed OverviewChart and dimension selector per user request

const App: React.FC = () => {
  const [selectedItem, setSelectedItem] = useState<string>('')
  const [selectedCategory, setSelectedCategory] = useState<string>('')
  const [rangeType, setRangeType] = useState<string>('week')
  const [fromDate, setFromDate] = useState<string>('')
  const [toDate, setToDate] = useState<string>('')
  // Forecast period selection
  const [forecastRangeType, setForecastRangeType] = useState<string>('week')
  const [forecastDays, setForecastDays] = useState<number>(7)
  // AI 모델 및 인사이트
  const [models, setModels] = useState<string[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [insight, setInsight] = useState<string>('')
  const [page, setPage] = useState<string>('status')
  const [loading, setLoading] = useState<boolean>(false)
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([])
  const [question, setQuestion] = useState<string>('')
  // For triggering manual refresh
  const [refreshKey, setRefreshKey] = useState<number>(0)

  useEffect(() => {
    if (rangeType === 'custom') return
    const now = new Date()
    const to = now.toISOString().split('T')[0]
    let from = to
    switch (rangeType) {
      case 'day':
        from = to
        break
      case 'week':
        from = new Date(now.getTime() - 6*24*60*60*1000).toISOString().split('T')[0]
        break
      case 'month': {
        const m = new Date(now.getFullYear(), now.getMonth()-1, now.getDate())
        from = m.toISOString().split('T')[0]
      } break
      case 'year': {
        const y = new Date(now.getFullYear()-1, now.getMonth(), now.getDate())
        from = y.toISOString().split('T')[0]
      } break
    }
    setFromDate(from)
    setToDate(to)
  }, [rangeType])

  useEffect(() => {
    switch (forecastRangeType) {
      case 'week': setForecastDays(7); break
      case 'month': setForecastDays(30); break
      case 'year': setForecastDays(365); break
      default: break
    }
  }, [forecastRangeType])

  // fetch available Ollama models
  useEffect(() => {
    // When refreshKey changes, trigger manual data reload on server
    axios.post('/api/refresh-data')
      .then(() => {
        // After server-side refresh, re-fetch available models
        return axios.get('/api/models')
      })
      .then(res => {
        setModels(res.data);
        if (res.data.length) setSelectedModel(res.data[0]);
      })
      .catch(err => console.error('Refresh error:', err));
  }, [refreshKey])

  // Automatic default insight fetch for status page
  useEffect(() => {
    if (page !== 'status') return;
    setLoading(true);
    axios.post('/api/insight', {
      item: selectedItem,
      category: selectedCategory,
      from_date: fromDate,
      to_date: toDate,
      model: selectedModel
    }).then(res => {
      setMessages([{ role: 'assistant', content: res.data.insight }]);
    }).catch(err => {
      console.error(err);
      setMessages([{ role: 'assistant', content: '기본 인사이트를 불러오는 중 오류가 발생했습니다.' }]);
    }).finally(() => setLoading(false));
  }, [page, selectedItem, selectedCategory, fromDate, toDate, selectedModel, refreshKey]);

  return (
    <div className="app-container">
      {/* 1. 페이지 제목 */}
      <div className="comp-item">
        <span className="comp-label">1</span>
        <h1>출고 수량 분석 대시보드</h1>
      </div>
      {/* 2. 데이터 새로고침 */}
      <div className="comp-item">
        <span className="comp-label">2</span>
        <button onClick={() => setRefreshKey(prev => prev + 1)}>데이터 새로고침</button>
      </div>
      {/* 3. 페이지 네비게이션 */}
      <div className="comp-item page-nav">
        <span className="comp-label">3</span>
        <button disabled={page === 'status'} onClick={() => setPage('status')}>현황 분석</button>
        <button disabled={page === 'forecast'} onClick={() => setPage('forecast')}>예측 분석</button>
      </div>

      {/* 현황 분석 페이지 */}
      {page === 'status' && (
        <>
          {/* 4. 기간 유형 선택 */}
          <div className="control-panel comp-item">
            <span className="comp-label">4</span>
            <label>기간 유형: </label>
            <select value={rangeType} onChange={e => setRangeType(e.target.value)}>
              <option value="day">일간</option>
              <option value="week">주간</option>
              <option value="month">월간</option>
              <option value="year">연간</option>
              <option value="custom">사용자 지정</option>
            </select>
            {rangeType === 'custom' && (
              <>
                <label htmlFor="from-date">시작 일자: </label>
                <input type="date" id="from-date" value={fromDate} onChange={e => setFromDate(e.target.value)} />
                <label htmlFor="to-date">종료 일자: </label>
                <input type="date" id="to-date" value={toDate} onChange={e => setToDate(e.target.value)} />
              </>
            )}
          </div>
          {/* 5. 아이템/분류 선택 */}
          <div className="comp-item">
            <ItemCategorySelector
              item={selectedItem}
              category={selectedCategory}
              onItemChange={setSelectedItem}
              onCategoryChange={setSelectedCategory}
            />
          </div>
          {/* 6. 트렌드 차트 */}
          <div className="chart-container">
            <TrendChart
              item={selectedItem}
              category={selectedCategory}
              fromDate={fromDate}
              toDate={toDate}
            />
          </div>
          {/* 7. AI 인사이트 챗봇 */}
          <div className="comp-item ai-container">
            <span className="comp-label">7</span>
            {loading ? (
              <p>인사이트 로딩 중...</p>
            ) : (
              <div className="chat-window">
                {messages.map((m, idx) => (
                  <div key={idx} className={`chat-message ${m.role}`}> 
                    <strong>{m.role === 'user' ? '사용자' : 'AI'}:</strong> {m.content}
                  </div>
                ))}
              </div>
            )}
            {/* 질문 입력 */}
            <div className="chat-input">
              <input type="text" value={question} placeholder="추가 질문을 입력하세요" onChange={e => setQuestion(e.target.value)} />
              <button disabled={loading || !question} onClick={async () => {
                const q = question;
                setLoading(true);
                setMessages(prev => [...prev, { role: 'user', content: q }]);
                setQuestion('');
                try {
                  const res = await axios.post('/api/insight', {
                    item: selectedItem,
                    category: selectedCategory,
                    from_date: fromDate,
                    to_date: toDate,
                    model: selectedModel,
                    question: q
                  });
                  setMessages(prev => [...prev, { role: 'assistant', content: res.data.insight }]);
                } catch (err) {
                  console.error(err);
                  setMessages(prev => [...prev, { role: 'assistant', content: '질문 처리 중 오류가 발생했습니다.' }]);
                } finally {
                  setLoading(false);
                }
              }}>질문</button>
            </div>
          </div>
        </>
      )}
      {/* 예측 분석 페이지 */}
      {page === 'forecast' && (
        <>
          {/* 8. 예측 페이지 헤더 */}
          <div className="comp-item">
            <span className="comp-label">8</span>
            <h2>예측 분석</h2>
          </div>
          {/* 9. 예측 기간 선택 */}
          <div className="control-panel comp-item">
            <span className="comp-label">9</span>
            <label>예측 기간 유형: </label>
            <select value={forecastRangeType} onChange={e => setForecastRangeType(e.target.value)}>
              <option value="week">1주일</option>
              <option value="month">1개월</option>
              <option value="year">1년</option>
              <option value="custom">직접 입력</option>
            </select>
            {forecastRangeType === 'custom' && (
              <input type="number" min={1} value={forecastDays} onChange={e => setForecastDays(Number(e.target.value))} />
            )}
          </div>
          {/* 10. 예측 차트 */}
          <div className="chart-container">
            <ForecastChart
              item={selectedItem}
              category={selectedCategory}
              periods={forecastDays}
              fromDate={fromDate}
              lastDate={toDate}
            />
          </div>
        </>
      )}
    </div>
  )
}

export default App