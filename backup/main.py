from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
from typing import Optional
from datetime import date, timedelta, datetime
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import subprocess
import logging
import sqlite3
import requests
import io

# Configure logger for debugging
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG)

from app.analysis import load_df, aggregate_dimension, aggregate_trend
from forecast import forecast_series, create_events_df, train_residual_model, predict_with_residual_correction, safe_forecast_series
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Load settings
load_dotenv()
db_path = os.getenv("DB_PATH", "vf.db")
table_name = os.getenv("TABLE_NAME", "vf 출고 수량 ocr google 보고서 - 일별 출고 수량 (4)")

# CSV URL for real-time data updates
CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQwqI0BG-d2aMrql7DK4fQQTjvu57VtToSLAkY_nq92a4Cg5GFVbIn6_IR7Fq6_O-2TloFSNlXT8ZWC/pub?gid=1152588885&single=true&output=csv"

REALTIME_CSV_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQYW_XI-stT0t4KqqpDW0DcBud_teV8223_vupnZsO3DrbqRqZkwXBplXSld8sB_qEXL92Ckn7J8B29/pub?gid=572466553&single=true&output=csv'
DB_PATH = 'your.db'  # adjust as needed

def fetch_csv_to_db():
    try:
        df_csv = pd.read_csv(CSV_URL)
        conn = sqlite3.connect(db_path)
        df_csv.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        logger.debug(f"Fetched CSV data ({df_csv.shape}) saved to {table_name}")
    except Exception as e:
        logger.error(f"Error fetching CSV data: {e}")

def download_and_store_realtime():
    resp = requests.get(REALTIME_CSV_URL)
    resp.raise_for_status()
    # Try utf-8-sig, then cp949, then fallback
    try:
        df = pd.read_csv(io.BytesIO(resp.content), encoding='utf-8-sig')
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(resp.content), encoding='cp949')
        except Exception:
            df = pd.read_csv(io.BytesIO(resp.content))
    df.columns = [str(c).strip() for c in df.columns]
    # If columns are field1, field2, ... use first row as header
    if all(str(col).startswith('field') for col in df.columns):
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        df.columns = [str(c).strip() for c in df.columns]
    # Auto-detect columns
    date_col = next((c for c in df.columns if '날짜' in c or '일자' in c), None)
    day_col = next((c for c in df.columns if '요일' in c), None)
    total_col = next((c for c in df.columns if '합계' in c), None)
    if not (date_col and day_col and total_col):
        raise Exception(f'필수 컬럼이 없습니다: {df.columns}')
    melt = df.melt(id_vars=[date_col, day_col, total_col], var_name='hour', value_name='shipment')
    melt['hour'] = pd.to_numeric(melt['hour'], errors='coerce')
    melt = melt.dropna(subset=['hour'])
    melt['hour'] = melt['hour'].astype(int)
    melt[date_col] = pd.to_datetime(melt[date_col], errors='coerce').dt.strftime('%Y-%m-%d')
    melt = melt.rename(columns={date_col: '날짜'})
    conn = sqlite3.connect(DB_PATH)
    melt.to_sql('realtime_shipments', conn, if_exists='replace', index=False)
    conn.close()

# Initialize application and data
app = FastAPI(title="출고 수량 분석 API")

# Add CORS middleware to allow external access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initial data load from CSV
fetch_csv_to_db()
df = load_df(db_path, table_name)
dimensions = aggregate_dimension(df)

# NEW: Download realtime data at startup
try:
    download_and_store_realtime()
except Exception as e:
    logger.error(f'Failed to download realtime data at startup: {e}')

# Scheduler to refresh data daily at midnight
def refresh_data():
    global df, dimensions
    # Refresh data from CSV and reload
    fetch_csv_to_db()
    df = load_df(db_path, table_name)
    dimensions = aggregate_dimension(df)

scheduler = AsyncIOScheduler()
scheduler.add_job(refresh_data, 'cron', hour=0) # Main historical data (daily)
scheduler.add_job(download_and_store_realtime, 'interval', hours=1) # Real-time hourly data (every hour)
scheduler.start()

# Models
class TrendParams(BaseModel):
    item: Optional[str] = None
    category: Optional[str] = None
    from_date: Optional[date] = None
    to_date: Optional[date] = None

class ForecastParams(BaseModel):
    item: Optional[str] = None
    category: Optional[str] = None
    periods: int = Field(30, gt=0)
    from_date: Optional[date] = None
    last_date: Optional[date] = None
    use_custom: bool = False
    freq: Optional[str] = None

class BacktestParams(BaseModel):
    item: Optional[str] = None
    category: Optional[str] = None
    from_date: Optional[date] = None
    to_date: Optional[date] = None

# Routes
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "출고 수량 분석 API",
        "overview_endpoint": "/api/overview",
        "trend_endpoint": "/api/trend",
        "forecast_endpoint": "/api/forecast",
        "docs": "/docs"
    }

@app.get("/api/overview")
def get_overview(dimension: str = Query("year", enum=["year", "month", "week", "period", "weekday"])):
    """Return aggregated sums by the specified time dimension."""
    return dimensions.get(dimension, [])

@app.post("/api/trend")
def get_trend(params: TrendParams):
    """Return daily trend filtered by item, category, and optional date range."""
    result = aggregate_trend(df, item=params.item, category=params.category, from_date=params.from_date, to_date=params.to_date)
    return result.to_dict(orient="records")

@app.post("/api/forecast")
def get_forecast(params: ForecastParams):
    """Return forecasted values for 수량(박스) filtered by item, category, and date range."""
    # 시간별 예측 요청 시, 최근 7일간 평균 시간별 출고량 누적으로 예측
    if getattr(params, 'freq', None) == 'H':
        logger.debug("Handling hourly forecast request based on avg increments")
        today_dt = pd.to_datetime(params.from_date or datetime.now().strftime('%Y-%m-%d'))
        # 실제 오늘 데이터
        conn = sqlite3.connect(DB_PATH)
        try:
            df_today = pd.read_sql_query(
                "SELECT hour, shipment FROM realtime_shipments WHERE 날짜 = ? ORDER BY hour",
                conn, params=[today_dt.strftime('%Y-%m-%d')]
            )
            logger.debug(f"Today's data (df_today) shape: {df_today.shape}\n{df_today.head()}")
        except Exception as e:
            logger.error(f"Error fetching today's data: {e}")
            df_today = pd.DataFrame({'hour':[], 'shipment':[]})
        # 과거 7일 데이터
        seven = (today_dt - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        try:
            df_hist = pd.read_sql_query(
                "SELECT 날짜, hour, shipment FROM realtime_shipments WHERE 날짜 >= ? AND 날짜 < ? ORDER BY 날짜, hour",
                conn, params=[seven, today_dt.strftime('%Y-%m-%d')]
            )
            logger.debug(f"History data (df_hist) shape: {df_hist.shape}\n{df_hist.head()}")
        except Exception as e:
            logger.error(f"Error fetching history data: {e}")
            df_hist = pd.DataFrame({'날짜':[], 'hour':[], 'shipment':[]})
        conn.close()

        # 오늘 데이터에서 실적이 있는 시간대 찾기
        present_hours = df_today.dropna(subset=['shipment'])['hour'].astype(int).tolist()
        first_missing_hour = next((h for h in range(24) if h not in present_hours), 24)
        forecastStart = first_missing_hour  # 실적이 없는 첫 시간부터 예측 시작

        # 누적 실적의 마지막 값을 예측의 시작점으로 사용
        if present_hours:
            last_actual_hour = max(present_hours)
            # shipment 값이 NaN일 경우 0으로 대체
            shipment_val = df_today[df_today['hour'] == last_actual_hour]['shipment'].values
            if len(shipment_val) > 0 and pd.notna(shipment_val[0]):
                last_actual_cum = int(shipment_val[0])
            else:
                # 마지막 유효한(숫자인) shipment 값 찾기
                valid_shipments = df_today.dropna(subset=['shipment'])
                if not valid_shipments.empty:
                    last_actual_cum = int(valid_shipments.iloc[-1]['shipment'])
                else:
                    last_actual_cum = 0
        else:
            last_actual_cum = 0

        logger.debug(f"실적 데이터가 있는 시간대: {present_hours}")
        logger.debug(f"실적이 없는 첫 시간: {forecastStart}")
        logger.debug(f"누적 실적 마지막 값: {last_actual_cum}")

        # 시간별 평균 *증가량* 계산
        df_hist['shipment'] = pd.to_numeric(df_hist['shipment'], errors='coerce')
        df_hist = df_hist.dropna(subset=['shipment']) # NaN shipment 값 제거
        df_hist = df_hist.sort_values(by=['날짜', 'hour'])
        df_hist['increment'] = df_hist.groupby('날짜')['shipment'].diff() # diff 후 NaN 가능성 있음 (각 날짜의 첫 시간)
        
        # 평균 증가량 계산 시 NaN을 0으로 채우고, 음수 증가량은 0으로 처리
        avg_inc_per_hour = df_hist.groupby('hour')['increment'].mean().fillna(0).clip(lower=0).to_dict()
        logger.debug(f"Average increment per hour: {avg_inc_per_hour}")

        # 예측 누적값 계산 (평균 증가량 기반)
        result = []
        pred_cum = last_actual_cum
        for h in range(forecastStart, 24):
            inc = avg_inc_per_hour.get(h, 0)
            if pd.isna(inc):
                inc = 0
            pred_cum += inc
            ds = today_dt + pd.Timedelta(hours=h)
            result.append({'ds': ds.strftime('%Y-%m-%dT%H:%M:%S'), 'yhat': int(round(pred_cum))})
        # 0~23시 전체에 대해 실적이 없는 시간대는 예측값, 실적이 있는 시간대는 None으로 채움
        full_result = []
        for h in range(24):
            ds = today_dt + pd.Timedelta(hours=h)
            if h in present_hours:
                full_result.append({'ds': ds.strftime('%Y-%m-%dT%H:%M:%S'), 'yhat': None})
            else:
                found = next((r for r in result if pd.to_datetime(r['ds']).hour == h), None)
                if found and found['yhat'] is not None:
                    full_result.append(found)
                else:
                    full_result.append({'ds': ds.strftime('%Y-%m-%dT%H:%M:%S'), 'yhat': 0})
        logger.debug(f"Final forecast result (hourly, full 0~23): {full_result}")
        return {'forecast': full_result}
    # 그 외 일별 예측
    df2 = df
    # Debug: log input parameters
    logger.debug(f"get_forecast called with: {params}")
    # Debug: initial data size
    logger.debug(f"Initial df size: {df2.shape}")
    # Filter by item and/or category
    if params.item:
        df2 = df2[df2['품목'] == params.item]
        logger.debug(f"Filtered by item '{params.item}', size: {df2.shape}")
    if params.category:
        df2 = df2[df2['분류'] == params.category]
        logger.debug(f"Filtered by category '{params.category}', size: {df2.shape}")
    # Create event flags for improved accuracy and residual correction
    start_date = df2['일자'].min()
    events_df = create_events_df(start_date, start_date + timedelta(days=params.periods))
    # Generate full forecast including historical and future
    forecast_full = safe_forecast_series(
        df2,
        '일자',
        '수량(박스)',
        periods=params.periods,
        freq=params.freq or 'D',
        use_custom=params.use_custom,
        events_df=events_df
    )
    logger.debug(f"Full forecast results sample:\n{forecast_full.head()}")
    if params.last_date:
        cutoff = pd.to_datetime(params.last_date)
        # Prepare actual historical data
        ts_hist = df2[['일자', '수량(박스)']].dropna().rename(columns={'일자':'ds','수량(박스)':'y'})
        ts_hist = ts_hist.groupby('ds')['y'].sum().reset_index()
        try:
            # Align predictions with actuals by merging on ds
            hist_pred = forecast_full[['ds', 'yhat']].merge(ts_hist, on='ds', how='inner')
            actuals = hist_pred['y']
            preds = hist_pred['yhat']
            # Compute error metrics (skip if empty)
            if len(actuals) > 0 and len(preds) > 0:
                mse = mean_squared_error(actuals, preds)
                mae = mean_absolute_error(actuals, preds)
                mape = mean_absolute_percentage_error(actuals, preds)
            else:
                mse = mae = mape = None
            # Generate future forecasts using last_date cutoff
            forecast_future = forecast_full[forecast_full['ds'] > cutoff].copy()
            # 잔차 보정도 데이터가 있을 때만
            if not hist_pred.empty and not forecast_future.empty:
                residual_model = train_residual_model(ts_hist, hist_pred[['ds','yhat']], events_df=events_df)
                corrected_df = predict_with_residual_correction(residual_model, forecast_future, events_df=events_df)
                forecast_future = forecast_future.merge(corrected_df, on='ds', how='left')
            return {
                'metrics': {'mse': mse, 'mae': mae, 'mape': mape},
                'forecast': forecast_future.to_dict(orient='records')
            }
        except Exception as e:
            logger.error(f"Residual correction failed, returning raw forecast. Error: {e}")
            # Fallback: return only future forecast values
            forecast_future = forecast_full[forecast_full['ds'] > cutoff].copy() if 'cutoff' in locals() else forecast_full
            return forecast_future.to_dict(orient='records')
    else:
        return forecast_full.to_dict(orient='records')

# Endpoint to get unique items (품목)
@app.get("/api/items")
def get_items(category: str = Query(None, description="분류 이름")):
    """Return list of unique items (품목), optionally filtered by category"""
    if category:
        items = df[df['분류'] == category]['품목'].dropna().unique().tolist()
    else:
        items = df['품목'].dropna().unique().tolist()
    return items

# Endpoint to get categories (분류) for a given item
@app.get("/api/categories")
def get_categories(item: str = Query(None, description="품목 이름")):
    """Return list of unique categories (분류), optionally filtered by item"""
    if item:
        cats = df[df['품목'] == item]['분류'].dropna().unique().tolist()
    else:
        cats = df['분류'].dropna().unique().tolist()
    return cats

@app.get("/api/models")
def get_models():
    """Return list of local Ollama models."""
    # static list; replace with dynamic `ollama list` if needed
    return ["gemma3:latest", "exaone-deep:7.8b", "gemma3:4b"]

class InsightParams(BaseModel):
    item: Optional[str] = None
    category: Optional[str] = None
    from_date: date
    to_date: date
    model: str
    question: Optional[str] = None  # optional follow-up question for chat

@app.post("/api/insight")
def get_insight(params: InsightParams):
    """Handle initial summary of top-performing categories and follow-up Q&A."""
    logger.debug(f"get_insight called with: {params}")
    # If no follow-up question, generate default summary of categories with most increase
    if params.question is None:
        # Filter by item if provided
        df2 = df.copy()
        if params.item:
            df2 = df2[df2['품목'] == params.item]
        # Current period sums by category
        df_curr = df2[(df2['일자'] >= pd.to_datetime(params.from_date)) & (df2['일자'] <= pd.to_datetime(params.to_date))]
        curr_sum = df_curr.groupby('분류')['수량(박스)'].sum()
        # Previous year same period
        prev_from = params.from_date - timedelta(days=365)
        prev_to = params.to_date - timedelta(days=365)
        df_prev = df2[(df2['일자'] >= pd.to_datetime(prev_from)) & (df2['일자'] <= pd.to_datetime(prev_to))]
        prev_sum = df_prev.groupby('분류')['수량(박스)'].sum()
        # Combine and compute percent change
        df_cat = pd.DataFrame({'curr': curr_sum, 'prev': prev_sum}).fillna(0)
        df_cat = df_cat[df_cat['prev'] > 0]
        if not df_cat.empty:
            # Compute percent changes and pick top 5 categories
            df_cat['pct'] = (df_cat['curr'] - df_cat['prev']) / df_cat['prev'] * 100
            top5 = df_cat.sort_values('pct', ascending=False).head(5)
            # Format category summaries
            cats = [f"{cat}({row.pct:+.1f}%↑)" for cat, row in top5.iterrows()]
            # For each top category, find top 5 items by percent change
            details = []
            for cat in top5.index:
                # current and previous for items within category
                df_curr_cat = df2[(df2['분류'] == cat) & (df2['일자'] >= pd.to_datetime(params.from_date)) & (df2['일자'] <= pd.to_datetime(params.to_date))]
                df_prev_cat = df2[(df2['분류'] == cat) & (df2['일자'] >= pd.to_datetime(params.from_date - timedelta(days=365))) & (df2['일자'] <= pd.to_datetime(params.to_date - timedelta(days=365)))]
                curr_items = df_curr_cat.groupby('품목')['수량(박스)'].sum()
                prev_items = df_prev_cat.groupby('품목')['수량(박스)'].sum()
                df_item = pd.DataFrame({'curr': curr_items, 'prev': prev_items}).fillna(0)
                df_item = df_item[df_item['prev'] > 0]
                if not df_item.empty:
                    df_item['pct'] = (df_item['curr'] - df_item['prev']) / df_item['prev'] * 100
                    top_items = df_item.sort_values('pct', ascending=False).head(5).index.tolist()
                    details.append(f"{cat}: {', '.join(top_items)}")
            default_summary = (
                f"이번 기간 수량이 가장 크게 증가한 상위 5개 분류는 {', '.join(cats)}입니다. "
                f"세부 품목별 상위 5개는 { '; '.join(details) }입니다."
            )
        else:
            default_summary = "이번 기간 급격한 증가를 보인 분류가 없습니다."
        return {"insight": default_summary}
    # Otherwise, handle follow-up with AI
    # Compute basic metrics for context
    df_curr = aggregate_trend(df, item=params.item, category=params.category,
                              from_date=params.from_date, to_date=params.to_date)
    curr_avg = df_curr['수량(박스)'].mean() or 0
    prev_from = params.from_date - timedelta(days=365)
    prev_to = params.to_date - timedelta(days=365)
    df_prev = aggregate_trend(df, item=params.item, category=params.category,
                              from_date=prev_from, to_date=prev_to)
    prev_avg = df_prev['수량(박스)'].mean() or 0
    pct = ((curr_avg - prev_avg) / prev_avg * 100) if prev_avg else 0
    emoji = "⬆️" if pct > 0 else ("⬇️" if pct < 0 else "➡️")
    # Build prompt including user question
    prompt = (
        f"현재 기간 평균 수량은 {curr_avg:.1f} 박스, 전년 동기 평균 수량은 {prev_avg:.1f} 박스입니다."
        f" 변화율은 {pct:+.1f}%이며, '{emoji}' 입니다."
        f" 추가 질문: {params.question}"
    )
    logger.debug(f"Insight prompt (Q&A): {prompt}")
    # Call ollama CLI
    result = subprocess.run(
        ["ollama", "run", params.model, prompt],
        capture_output=True, text=True, encoding='utf-8', errors='replace'
    )
    logger.debug(f"ollama returncode: {result.returncode}")
    if result.returncode == 0:
        out = result.stdout or ""
        logger.debug(f"ollama stdout: {out}")
        summary = out.strip() or "AI 인사이트 생성은 성공했으나, 출력이 없습니다."
    else:
        logger.error(f"ollama stderr: {result.stderr}")
        err_msg = result.stderr.strip()
        summary = err_msg or f"AI 생성 실패 (exit code {result.returncode})"
    return {"insight": summary}

# Endpoint to manually refresh data from CSV to DB
@app.post("/api/refresh-data")
def refresh_data_endpoint():
    """Fetch the remote CSV, replace DB table, and reload in-memory data."""
    fetch_csv_to_db()
    global df, dimensions
    df = load_df(db_path, table_name)
    dimensions = aggregate_dimension(df)
    return {"status": "data refreshed"}

@app.post("/api/backtest")
def get_backtest(params: BacktestParams):
    """Return historical daily forecast and error rate between from_date and to_date."""
    df2 = df
    if params.item:
        df2 = df2[df2['품목'] == params.item]
    if params.category:
        df2 = df2[df2['분류'] == params.category]
    if params.from_date:
        df2 = df2[df2['일자'] >= pd.to_datetime(params.from_date)]
    if params.to_date:
        df2 = df2[df2['일자'] <= pd.to_datetime(params.to_date)]
    # Get forecast series including historical dates
    hist_fc = forecast_series(df2, '일자', '수량(박스)', periods=0, freq='D')
    # Prepare actuals
    actual = df2[['일자', '수량(박스)']].dropna().rename(columns={'일자':'ds', '수량(박스)':'y'})
    actual = actual.groupby('ds')['y'].sum().reset_index()
    # Merge and compute error rate
    merged = hist_fc.merge(actual, on='ds', how='inner')
    merged['error_rate'] = (merged['yhat'] - merged['y']).abs() / merged['y'] * 100
    # Return per-day records
    return merged[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper', 'error_rate']].to_dict(orient='records')

@app.post('/api/realtime/refresh')
def refresh_realtime():
    download_and_store_realtime()  # 동기 실행
    return {'status': 'refresh completed'}

@app.get('/api/realtime/today')
def get_today_realtime():
    today = datetime.now().strftime('%Y-%m-%d')
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT hour, shipment FROM realtime_shipments WHERE 날짜 = ? ORDER BY hour",
            conn, params=(today,))
    except Exception as e:
        logger.warning(f'realtime_shipments table missing, attempting to download: {e}')
        download_and_store_realtime()
        df = pd.read_sql_query(
            "SELECT hour, shipment FROM realtime_shipments WHERE 날짜 = ? ORDER BY hour",
            conn, params=(today,))
    finally:
        conn.close()
    # Fill missing hours with None, and convert NaN to None
    result = [None]*24
    for _, row in df.iterrows():
        h = int(row['hour'])
        val = row['shipment']
        if pd.isna(val):
            result[h] = None
        else:
            result[h] = float(val)
    return {'date': today, 'shipments': result}

@app.get('/api/realtime/history')
def get_realtime_history(date: str):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT hour, shipment FROM realtime_shipments WHERE 날짜 = ? ORDER BY hour",
        conn, params=(date,))
    conn.close()
    result = [None]*24
    for _, row in df.iterrows():
        h = int(row['hour'])
        val = row['shipment']
        if pd.isna(val):
            result[h] = None
        else:
            result[h] = float(val)
    return {'date': date, 'shipments': result}

@app.get('/api/realtime/weekday-trend')
def get_weekday_trend():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT 날짜, hour, shipment FROM realtime_shipments", conn)
    conn.close()
    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    df = df.dropna(subset=['날짜'])
    df['weekday'] = df['날짜'].dt.weekday  # Monday=0, Sunday=6
    # 최근 4주만 사용
    max_date = df['날짜'].max()
    min_date = max_date - pd.Timedelta(days=28)
    df_recent = df[df['날짜'] >= min_date]
    # 요일별, 시간별 평균
    weekday_map = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
    result = {}
    for wd in range(7):
        arr = [None]*24
        for h in range(24):
            vals = df_recent[(df_recent['weekday']==wd) & (df_recent['hour']==h)]['shipment']
            arr[h] = float(vals.mean()) if not vals.empty else None
        result[weekday_map[wd]] = arr 