from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
from typing import Optional
from datetime import date, timedelta
from dotenv import load_dotenv
import pandas as pd
import subprocess
import logging
import sqlite3

# Configure logger for debugging
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG)

from app.analysis import load_df, aggregate_dimension, aggregate_trend
from forecast import forecast_series
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Load settings
load_dotenv()
db_path = os.getenv("DB_PATH", "vf.db")
table_name = os.getenv("TABLE_NAME", "vf 출고 수량 ocr google 보고서 - 일별 출고 수량 (4)")

# CSV URL for real-time data updates
CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQwqI0BG-d2aMrql7DK4fQQTjvu57VtToSLAkY_nq92a4Cg5GFVbIn6_IR7Fq6_O-2TloFSNlXT8ZWC/pub?gid=1152588885&single=true&output=csv"

def fetch_csv_to_db():
    try:
        df_csv = pd.read_csv(CSV_URL)
        conn = sqlite3.connect(db_path)
        df_csv.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        logger.debug(f"Fetched CSV data ({df_csv.shape}) saved to {table_name}")
    except Exception as e:
        logger.error(f"Error fetching CSV data: {e}")

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

# Scheduler to refresh data daily at midnight
def refresh_data():
    global df, dimensions
    # Refresh data from CSV and reload
    fetch_csv_to_db()
    df = load_df(db_path, table_name)
    dimensions = aggregate_dimension(df)

scheduler = AsyncIOScheduler()
scheduler.add_job(refresh_data, 'cron', hour=0)
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
    # Filter by historical period
    if params.from_date:
        df2 = df2[df2['일자'] >= pd.to_datetime(params.from_date)]
        logger.debug(f"Filtered by from_date '{params.from_date}', size: {df2.shape}")
    if params.last_date:
        df2 = df2[df2['일자'] <= pd.to_datetime(params.last_date)]
        logger.debug(f"Filtered by last_date '{params.last_date}', size: {df2.shape}")
    # Generate forecast
    forecast_df = forecast_series(df2, '일자', '수량(박스)', periods=params.periods, use_custom=params.use_custom)
    # Debug: show sample of forecast results
    logger.debug(f"Forecast results sample:\n{forecast_df.head()}")
    # Return only future dates beyond last_date
    if params.last_date:
        forecast_df = forecast_df[forecast_df['ds'] > pd.to_datetime(params.last_date)]
    return forecast_df.to_dict(orient="records")

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