import numpy as np, pandas as pd

def synthetic_demand(seed=42):
    """4 years of daily electricity demand (MWh): trend + weekly + yearly (summer AC / winter heating) + holidays + noise."""
    rng = np.random.default_rng(seed)
    d = pd.date_range("2021-01-01", "2024-12-31", freq="D")
    t = np.arange(len(d))
    yearly = 120 * np.cos(2 * np.pi * (d.dayofyear - 200) / 365.25) + 40 * np.cos(4 * np.pi * (d.dayofyear - 15) / 365.25)
    weekly = np.where(d.dayofweek >= 5, -110, 25) + 15 * np.sin(2 * np.pi * d.dayofweek / 7)
    holiday = np.where(((d.month == 12) & (d.day >= 24)) | ((d.month == 1) & (d.day == 1)), -90, 0)
    y = 1500 + 0.25 * t + yearly + weekly + holiday + rng.normal(0, 28, len(d))
    return pd.DataFrame({"ds": d, "y": y})

def load(csv=None, date_col="ds", target_col="y"):
    if csv is None: return synthetic_demand()
    df = pd.read_csv(csv, parse_dates=[date_col]).rename(columns={date_col: "ds", target_col: "y"})[["ds", "y"]]
    df = df.sort_values("ds").set_index("ds").asfreq("D").interpolate().reset_index()
    return df
