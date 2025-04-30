import sqlite3
import pandas as pd
import numpy as np
from prophet.diagnostics import cross_validation, performance_metrics
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
try:
    from tbats import TBATS
except ImportError:
    TBATS = None
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
try:
    from prophet import Prophet
except ImportError:
    try:
        from fbprophet import Prophet
    except ImportError:
        raise ImportError("Prophet library is not installed. Please install prophet or fbprophet.")
from statsmodels.tsa.statespace.sarimax import SARIMAX
import lightgbm as lgb
import schedule
import time

def load_df(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f'SELECT * FROM "{table_name}";', conn)
    conn.close()
    # Rename generic headers
    if all(str(col).startswith('field') for col in df.columns) and not df.empty:
        header = df.iloc[0].tolist()
        df = df.iloc[1:].reset_index(drop=True)
        df.columns = header
    # Clean and cast
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    df['일자'] = pd.to_datetime(df['일자'], errors='coerce')
    df['수량(박스)'] = pd.to_numeric(df['수량(박스)'].astype(str).str.replace(',', ''), errors='coerce')
    # Convert 판매금액 to numeric if exists
    if '판매금액' in df.columns:
        df['판매금액'] = pd.to_numeric(df['판매금액'].astype(str).str.replace(',', ''), errors='coerce')
    # Ensure 품목, 분류 columns exist
    return df

def forecast_series(df, date_col, value_col, periods=30, freq='D', use_custom: bool = False, exog_cols=None, events_df=None):
    ts = df[[date_col, value_col]].dropna().rename(columns={date_col: 'ds', value_col: 'y'})
    ts = ts.groupby('ds')['y'].sum().reset_index()
    # Feature engineering: one-hot weekday
    ts['weekday'] = ts['ds'].dt.weekday
    dow_dummies = pd.get_dummies(ts['weekday'], prefix='dow')
    ts = pd.concat([ts, dow_dummies], axis=1)
    # Feature engineering: one-hot period of month
    ts['period_code'] = ts['ds'].dt.day.apply(lambda x: 1 if x <= 10 else (2 if x <= 20 else 3))
    period_dummies = pd.get_dummies(ts['period_code'], prefix='period')
    ts = pd.concat([ts, period_dummies], axis=1)
    # Feature engineering: one-hot season labels
    def get_season(month):
        if month in [3,4,5]: return 'spring'
        elif month in [6,7,8]: return 'summer'
        elif month in [9,10,11]: return 'autumn'
        else: return 'winter'
    ts['season'] = ts['ds'].dt.month.apply(get_season)
    season_dummies = pd.get_dummies(ts['season'], prefix='season')
    ts = pd.concat([ts, season_dummies], axis=1)
    # Merge exogenous variables if provided
    if exog_cols:
        exog_df = df[[date_col] + exog_cols].copy()
        exog_df.rename(columns={date_col: 'ds'}, inplace=True)
        exog_agg = exog_df.groupby('ds')[exog_cols].sum().reset_index()
        ts = ts.merge(exog_agg, on='ds', how='left').fillna(0)
    # Merge event flags if provided
    if events_df is not None:
        ts = ts.merge(events_df, on='ds', how='left').fillna(0)
    # Initialize Prophet model; apply custom seasonalities and prior scales if requested
    if use_custom:
        # Define changepoints for surge (Sep 1) and decline (Mar 15)
        years = ts['ds'].dt.year.unique()
        changepoints = []
        for y in years:
            changepoints.append(pd.to_datetime(f"{y}-09-01"))
            changepoints.append(pd.to_datetime(f"{y}-03-15"))
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                        changepoints=changepoints, changepoint_prior_scale=0.5, seasonality_prior_scale=10.0)
        # Add custom seasonalities
        model.add_seasonality(name='weekly', period=7, fourier_order=10)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=10)
        model.add_seasonality(name='quarterly', period=365.25/4, fourier_order=5)
        # Add regressors for all engineered features and exogenous variables
        for reg_col in [c for c in ts.columns if c not in ['ds', 'y']]:
            model.add_regressor(reg_col)
    else:
        model = Prophet()
    # Fit model including regressors if custom
    model.fit(ts)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    # Feature engineering for future dataframe
    future['weekday'] = future['ds'].dt.weekday
    dow_f = pd.get_dummies(future['weekday'], prefix='dow')
    future = pd.concat([future, dow_f], axis=1)
    future['period_code'] = future['ds'].dt.day.apply(lambda x: 1 if x <= 10 else (2 if x <= 20 else 3))
    period_f = pd.get_dummies(future['period_code'], prefix='period')
    future = pd.concat([future, period_f], axis=1)
    future['season'] = future['ds'].dt.month.apply(get_season)
    season_f = pd.get_dummies(future['season'], prefix='season')
    future = pd.concat([future, season_f], axis=1)
    # Fill future exogenous features with zeros or user-provided values
    if exog_cols:
        for col in exog_cols:
            future[col] = 0
    # Merge future event flags if provided
    if events_df is not None:
        future = future.merge(events_df, on='ds', how='left').fillna(0)
    forecast = model.predict(future)
    # Round predictions to integers
    forecast['yhat'] = forecast['yhat'].round().astype(int)
    forecast['yhat_lower'] = forecast['yhat_lower'].round().astype(int)
    forecast['yhat_upper'] = forecast['yhat_upper'].round().astype(int)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def prophet_grid_search(ts, param_grid, cv_params):
    """Grid search over Prophet hyperparameters using cross-validation."""
    best_params = None
    best_score = float('inf')
    for cps in param_grid.get('changepoint_prior_scale', [0.05]):
        for sps in param_grid.get('seasonality_prior_scale', [10.0]):
            for fo in param_grid.get('fourier_order', [10]):
                model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                                changepoint_prior_scale=cps, seasonality_prior_scale=sps)
                model.add_seasonality(name='monthly', period=30.5, fourier_order=fo)
                model.fit(ts)
                df_cv = cross_validation(model,
                                         initial=cv_params['initial'],
                                         period=cv_params['period'],
                                         horizon=cv_params['horizon'])
                df_p = performance_metrics(df_cv)
                mape = df_p['mape'].mean()
                print(f"Tested cps={cps}, sps={sps}, fo={fo}, mean_mape={mape:.4f}")
                if mape < best_score:
                    best_score = mape
                    best_params = {'changepoint_prior_scale': cps,
                                   'seasonality_prior_scale': sps,
                                   'fourier_order': fo}
    print(f"Best params: {best_params} with mean MAPE {best_score:.4f}")
    return best_params

def decompose_ts(ts, model_type='additive', period=365):
    """Perform time series decomposition and return result."""
    result = seasonal_decompose(ts.set_index('ds')['y'], model=model_type, period=period)
    return result

def evaluate_prophet_model(model, initial, period, horizon):
    """Evaluate Prophet model using rolling-origin cross-validation."""
    df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
    df_p = performance_metrics(df_cv)
    print(df_p[['horizon', 'rmse', 'mae', 'mape']])
    return df_p

# Add event creation, SARIMAX forecast, residual correction, and ensemble functions
def create_events_df(start_date, end_date, freq='D'):
    """Create a DataFrame of event flags: move_season, new_term, covid_lockdown."""
    dates = pd.date_range(start_date, end_date, freq=freq)
    events = pd.DataFrame({'ds': dates})
    # Move season: Mar-May and Aug-Oct
    events['move_season'] = events['ds'].dt.month.isin([3,4,5,8,9,10]).astype(int)
    # New academic term: Mar & Sep
    events['new_term'] = events['ds'].dt.month.isin([3,9]).astype(int)
    # COVID lockdown: example period
    events['covid_lockdown'] = events['ds'].between('2020-02-20', '2020-05-31').astype(int)
    return events


def forecast_sarimax(df, date_col, value_col, order=(1,1,1), seasonal_order=(1,1,1,12), periods=30, freq='D', exog_cols=None):
    """Fit a SARIMAX model and forecast."""
    ts = df[[date_col, value_col]].dropna().rename(columns={date_col: 'ds', value_col: 'y'})
    ts = ts.groupby('ds')['y'].sum().reset_index().sort_values('ds')
    exog = None
    if exog_cols:
        ex = df[[date_col] + exog_cols].rename(columns={date_col: 'ds'})
        exog = ex.groupby('ds')[exog_cols].sum().reindex(ts['ds'], fill_value=0)
    model = SARIMAX(ts['y'], exog=exog, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    future_idx = pd.date_range(ts['ds'].iloc[-1] + pd.Timedelta(days=1), periods=periods, freq=freq)
    exog_future = None
    if exog_cols:
        exog_future = pd.DataFrame(0, index=future_idx, columns=exog_cols)
    pred = res.get_forecast(steps=periods, exog=exog_future)
    mean = pred.predicted_mean
    ci = pred.conf_int()
    df_fc = pd.DataFrame({
        'ds': future_idx,
        'yhat': mean.astype(int),
        'yhat_lower': ci.iloc[:, 0].astype(int),
        'yhat_upper': ci.iloc[:, 1].astype(int)
    }).reset_index(drop=True)
    return df_fc


def train_residual_model(ts_df, forecast_df, exog_cols=None, events_df=None):
    """Train a LightGBM model on residuals between actual and forecast."""
    # Merge actual and predicted, compute residual
    df = ts_df.merge(forecast_df[['ds', 'yhat']], on='ds')
    df['residual'] = df['y'] - df['yhat']
    # Features: weekday, period and season
    df['weekday'] = df['ds'].dt.weekday
    df['period_code'] = df['ds'].dt.day.apply(lambda x: 1 if x <= 10 else (2 if x <= 20 else 3))
    df['season'] = df['ds'].dt.month.apply(lambda m: 'spring' if m in [3,4,5] else 'summer' if m in [6,7,8] else 'autumn' if m in [9,10,11] else 'winter')
    # Create dummy features
    dow = pd.get_dummies(df['weekday'], prefix='dow')
    period = pd.get_dummies(df['period_code'], prefix='period')
    season = pd.get_dummies(df['season'], prefix='season')
    X = pd.concat([dow, period, season], axis=1)
    # Merge event flags
    if events_df is not None:
        df = df.merge(events_df, on='ds', how='left').fillna(0)
        event_cols = [c for c in events_df.columns if c != 'ds']
        X = pd.concat([X, df[event_cols]], axis=1)
    y = df['residual']
    model = lgb.LGBMRegressor()
    model.fit(X, y)
    model.feature_names_ = X.columns.tolist()
    return model


def predict_with_residual_correction(model, forecast_df, events_df=None):
    """Apply residual correction to forecast."""
    df = forecast_df.copy()
    # Create dummy features
    df['weekday'] = df['ds'].dt.weekday
    df['period_code'] = df['ds'].dt.day.apply(lambda x: 1 if x <= 10 else (2 if x <= 20 else 3))
    df['season'] = df['ds'].dt.month.apply(lambda m: 'spring' if m in [3,4,5] else 'summer' if m in [6,7,8] else 'autumn' if m in [9,10,11] else 'winter')
    X = pd.concat([
        pd.get_dummies(df['weekday'], prefix='dow'),
        pd.get_dummies(df['period_code'], prefix='period'),
        pd.get_dummies(df['season'], prefix='season')
    ], axis=1)
    # Merge event flags
    if events_df is not None:
        df = df.merge(events_df, on='ds', how='left').fillna(0)
        event_cols = [c for c in events_df.columns if c != 'ds']
        X = pd.concat([X, df[event_cols]], axis=1)
    # Align with model features
    X = X[model.feature_names_]
    df['residual_pred'] = model.predict(X)
    df['yhat_corrected'] = (df['yhat'] + df['residual_pred']).round().astype(int)
    return df[['ds', 'yhat_corrected']]


def ensemble_forecasts(forecasts: dict, weights: dict):
    """Combine multiple forecast DataFrames using specified weights."""
    # assume all forecasts have same ds
    ensemble = pd.DataFrame({'ds': forecasts[next(iter(forecasts))]['ds']})
    # weighted average
    yhat = sum(weights[k] * forecasts[k]['yhat'] for k in forecasts)
    lower = sum(weights[k] * forecasts[k]['yhat_lower'] for k in forecasts)
    upper = sum(weights[k] * forecasts[k]['yhat_upper'] for k in forecasts)
    ensemble['yhat'] = yhat.round().astype(int)
    ensemble['yhat_lower'] = lower.round().astype(int)
    ensemble['yhat_upper'] = upper.round().astype(int)
    return ensemble

def forecast_ets(df, date_col, value_col, periods=30, seasonal_periods=365, freq='D'):
    """Fit an ETS model and forecast."""
    ts = df[[date_col, value_col]].dropna().rename(columns={date_col: 'ds', value_col: 'y'})
    ts = ts.groupby('ds')['y'].sum().reset_index().sort_values('ds')
    model = ExponentialSmoothing(ts['y'], trend='add', seasonal='add', seasonal_periods=seasonal_periods).fit()
    future_idx = pd.date_range(ts['ds'].iloc[-1] + pd.Timedelta(days=1), periods=periods, freq=freq)
    pred = model.forecast(periods)
    df_fc = pd.DataFrame({'ds': future_idx, 'yhat': pred.round().astype(int)})
    df_fc['yhat_lower'] = df_fc['yhat']
    df_fc['yhat_upper'] = df_fc['yhat']
    return df_fc

def forecast_tbats(df, date_col, value_col, periods=30, freq='D'):
    """Fit a TBATS model and forecast."""
    if TBATS is None:
        raise ImportError("TBATS library not installed, skipping TBATS forecast.")
    ts = df[[date_col, value_col]].dropna().rename(columns={date_col: 'ds', value_col: 'y'})
    ts = ts.groupby('ds')['y'].sum().reset_index().sort_values('ds')
    estimator = TBATS(seasonal_periods=[7, 365.25])
    model = estimator.fit(ts['y'])
    future_idx = pd.date_range(ts['ds'].iloc[-1] + pd.Timedelta(days=1), periods=periods, freq=freq)
    y_forecast, conf_int = model.forecast(steps=periods, confidence_level=0.95)
    df_fc = pd.DataFrame({
        'ds': future_idx,
        'yhat': np.round(y_forecast).astype(int),
        'yhat_lower': np.round(conf_int[:,0]).astype(int),
        'yhat_upper': np.round(conf_int[:,1]).astype(int)
    })
    return df_fc

def detect_drift(ts_df, forecast_df, forecast_col='yhat', window=30, threshold=0.2):
    """Detect drift by computing MAPE over recent window for given forecast column."""
    df = ts_df.rename(columns={'y':'actual'}).merge(
        forecast_df[['ds', forecast_col]].rename(columns={forecast_col:'pred'}), on='ds')
    df = df.sort_values('ds').tail(window)
    mape_score = mean_absolute_percentage_error(df['actual'], df['pred'])
    print(f"Recent {window}-day MAPE on {forecast_col}: {mape_score:.3f}")
    return mape_score > threshold

def main():
    db_path = 'vf.db'
    table = 'vf 출고 수량 ocr google 보고서 - 일별 출고 수량 (4)'
    print(f"Loading data from {table}")
    df = load_df(db_path, table)
    # Prepare exogenous variable (average price per box)
    df['price_per_box'] = df['판매금액'] / df['수량(박스)']
    exog_cols = ['price_per_box']
    # Create event flags DataFrame from existing data
    min_date = df['일자'].min()
    max_date = df['일자'].max() + pd.Timedelta(days=30)
    events_df = create_events_df(min_date, max_date)
    # Prepare time series data for overall series
    ts_all = df[['일자', '수량(박스)']].dropna().rename(columns={'일자':'ds', '수량(박스)':'y'})
    ts_all = ts_all.groupby('ds')['y'].sum().reset_index()
    # Decompose series to inspect patterns
    print("Decomposing overall series:")
    decompose_ts(ts_all, period=30)
    # Prophet hyperparameter grid search
    print("Performing hyperparameter grid search for Prophet:")
    param_grid = {'changepoint_prior_scale': [0.1, 0.5, 1.0],
                  'seasonality_prior_scale': [1.0, 10.0, 20.0],
                  'fourier_order': [5, 10, 15]}
    cv_params = {'initial': '365 days', 'period': '180 days', 'horizon': '30 days'}
    best_params = prophet_grid_search(ts_all, param_grid, cv_params)
    # Forecast with best parameters
    print("Forecasting overall daily 수량(박스) with best parameters for next 30 days...")
    forecast_all = forecast_series(df, '일자', '수량(박스)', periods=30, freq='D', use_custom=True, exog_cols=exog_cols, events_df=events_df)
    # Evaluate best model
    print("Evaluating Prophet model with best parameters:")
    model_best = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                         **{k: best_params[k] for k in ['changepoint_prior_scale', 'seasonality_prior_scale']})
    model_best.add_seasonality(name='monthly', period=30.5, fourier_order=best_params['fourier_order'])
    model_best.fit(ts_all)
    evaluate_prophet_model(model_best, **cv_params)
    print(forecast_all.tail())
    # SARIMAX forecast
    sarimax_fc = forecast_sarimax(df, '일자', '수량(박스)', order=(1,1,1), seasonal_order=(1,1,1,12), periods=30, freq='D', exog_cols=exog_cols)
    print("SARIMAX Forecast:")
    print(sarimax_fc.tail())
    # ETS forecast
    ets_fc = forecast_ets(df, '일자', '수량(박스)', periods=30, seasonal_periods=30)
    print("ETS Forecast:")
    print(ets_fc.tail())
    # TBATS forecast (if available)
    if TBATS is not None:
        try:
            tbats_fc = forecast_tbats(df, '일자', '수량(박스)', periods=30)
            print("TBATS Forecast:")
            print(tbats_fc.tail())
        except Exception as e:
            print(f"TBATS forecast error: {e}")
            tbats_fc = forecast_all[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    else:
        print("TBATS library not installed, skipping TBATS forecast.")
        tbats_fc = forecast_all[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    # Residual correction with LightGBM
    resid_model = train_residual_model(ts_all, forecast_all, events_df=events_df)
    prophet_corr = predict_with_residual_correction(resid_model, forecast_all, events_df=events_df)
    print("Prophet Residual-Corrected Forecast:")
    print(prophet_corr.tail())
    # Ensemble forecasts including ETS and TBATS
    forecasts = {
        'prophet': forecast_all,
        'sarimax': sarimax_fc,
        'ets': ets_fc,
        'tbats': tbats_fc,
        'prophet_corr': prophet_corr
    }
    weights = {
        'prophet': 0.3,
        'sarimax': 0.2,
        'ets': 0.2,
        'tbats': 0.2,
        'prophet_corr': 0.1
    }
    ensemble_fc = ensemble_forecasts(forecasts, weights)
    print("Ensembled Forecast:")
    print(ensemble_fc.tail())
    # Detect drift
    if detect_drift(ts_all, ensemble_fc, forecast_col='yhat'):
        print("Drift detected: consider retraining the model.")
    # Example: forecast for a specific 품목 and 분류
    example_item = df['품목'].dropna().unique()[0]
    example_category = df[df['품목']==example_item]['분류'].iloc[0]
    print(f"Forecasting for item '{example_item}', category '{example_category}'")
    df_sub = df[(df['품목']==example_item) & (df['분류']==example_category)]
    forecast_sub = forecast_series(df_sub, '일자', '수량(박스)', periods=30, freq='D')
    print(forecast_sub.tail())

if __name__ == '__main__':
    main()
    # Schedule daily automatic runs
    schedule.every().day.at("00:00").do(main)
    while True:
        schedule.run_pending()
        time.sleep(60) 