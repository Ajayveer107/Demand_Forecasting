import time
import numpy as np, pandas as pd
from models import MODELS

def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {"MAE": np.mean(np.abs(y - p)), "RMSE": np.sqrt(np.mean((y - p) ** 2)),
            "MAPE%": np.mean(np.abs((y - p) / y)) * 100, "sMAPE%": np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p))) * 100}

def rolling_origin(df, horizon=14, n_folds=6, models=None):
    """Expanding-window backtest: each fold trains on everything before the cut-off and forecasts the next `horizon` days.
    No future data ever leaks into training."""
    models = models or MODELS
    n = len(df); rows, preds = [], {}
    for f in range(n_folds):
        cut = n - horizon * (n_folds - f)
        train, test = df.iloc[:cut], df.iloc[cut:cut + horizon]
        for name, fn in models.items():
            t0 = time.time()
            try: p = np.asarray(fn(train, horizon), float)
            except Exception as e: print(f"  skip {name}: {type(e).__name__}: {e}"); continue
            rows.append({"fold": f + 1, "model": name, **metrics(test["y"].values, p), "sec": time.time() - t0})
            preds.setdefault(name, []).append((test["ds"].values, p))
        print(f"fold {f + 1}/{n_folds} done", flush=True)
    return pd.DataFrame(rows), preds, n_folds
