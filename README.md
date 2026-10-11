# Time-Series Forecasting: Statistical vs ML vs Deep Learning
Forecasts daily electricity demand and compares **Seasonal Naive, SARIMA (statsmodels), Prophet, XGBoost (lag features) and LSTM (Keras)**
using **rolling-origin (expanding-window) backtesting** so no future data leaks into training.

## Run
    pip install -r requirements.txt
    python run.py                                   # built-in synthetic demand data
    python run.py --csv my.csv --date Date --target Sales --horizon 14 --folds 6
    python run.py --models "Seasonal Naive,Prophet,XGBoost"   # subset (faster)
Outputs: RESULTS.md (mean metrics), fold_results.csv (per fold), forecast.png.

## Method
- Metrics: MAE, RMSE, MAPE, sMAPE; every model is scored against a seasonal-naive baseline ("vs naive").
- XGBoost: lags (1-28d), rolling means, calendar + Fourier day-of-year features, recursive multi-step forecasting.
- LSTM: 28-day window, standardised, early stopping, recursive forecasting.

## Use real data
Electricity: UCI "ElectricityLoadDiagrams", or Kaggle "Hourly Energy Consumption (PJM)" (resample to daily).
Sales: Kaggle "Store Item Demand Forecasting" / "Rossmann". Stock volume: yfinance.
