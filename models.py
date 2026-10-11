"""Every model exposes forecast(train_df, horizon) -> np.ndarray. Missing optional libs are skipped gracefully."""
import warnings, os
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

def seasonal_naive(train, h, m=7):
    y = train["y"].values
    return np.array([y[-m + (i % m)] for i in range(h)])

def sarima(train, h):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = train["y"].values[-730:]  # last 2 years keeps it fast
    res = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False, maxiter=60)
    return res.forecast(h)

def prophet(train, h):
    import logging; logging.getLogger("cmdstanpy").setLevel(logging.ERROR); logging.getLogger("prophet").setLevel(logging.ERROR)
    from prophet import Prophet
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False).fit(train[["ds", "y"]])
    fut = m.make_future_dataframe(h, freq="D")
    return m.predict(fut)["yhat"].values[-h:]

# ---------- XGBoost: lag/rolling/calendar features, recursive multi-step ----------
LAGS = [1, 2, 3, 7, 14, 21, 28]
def _feats(hist, date):
    f = {f"lag{l}": hist[-l] for l in LAGS}
    f["roll7"], f["roll28"] = np.mean(hist[-7:]), np.mean(hist[-28:])
    f["dow"], f["month"] = date.dayofweek, date.month
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * date.dayofyear / 365.25), np.cos(2 * np.pi * date.dayofyear / 365.25)
    return f

def xgboost(train, h):
    from xgboost import XGBRegressor
    y, ds = train["y"].values, pd.to_datetime(train["ds"]).values
    X = pd.DataFrame([_feats(y[:i], pd.Timestamp(ds[i])) for i in range(28, len(y))])
    mdl = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0).fit(X, y[28:])
    hist, out, d = list(y), [], pd.Timestamp(ds[-1])
    for _ in range(h):
        d += pd.Timedelta(days=1)
        p = float(mdl.predict(pd.DataFrame([_feats(np.array(hist), d)]))[0]); out.append(p); hist.append(p)
    return np.array(out)

# ---------- LSTM: 28-day window, scaled, recursive ----------
def lstm(train, h, window=28, epochs=25):
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(0); np.random.seed(0)
    y = train["y"].values[-900:]
    mu, sd = y.mean(), y.std()
    z = (y - mu) / sd
    X = np.array([z[i:i + window] for i in range(len(z) - window)])[..., None]
    t = z[window:]
    model = keras.Sequential([keras.layers.Input((window, 1)), keras.layers.LSTM(48), keras.layers.Dense(1)])
    model.compile("adam", "mse")
    model.fit(X, t, epochs=epochs, batch_size=32, verbose=0, validation_split=0.1,
              callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
    win, out = list(z[-window:]), []
    for _ in range(h):
        p = float(model.predict(np.array(win[-window:])[None, :, None], verbose=0)[0, 0]); out.append(p); win.append(p)
    return np.array(out) * sd + mu

MODELS = {"Seasonal Naive": seasonal_naive, "SARIMA": sarima, "Prophet": prophet, "XGBoost": xgboost, "LSTM": lstm}
