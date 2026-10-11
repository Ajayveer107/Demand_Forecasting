import argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import load
from models import MODELS
from backtest import rolling_origin

ap = argparse.ArgumentParser()
ap.add_argument("--csv"); ap.add_argument("--date", default="ds"); ap.add_argument("--target", default="y")
ap.add_argument("--horizon", type=int, default=14); ap.add_argument("--folds", type=int, default=6)
ap.add_argument("--models", default=",".join(MODELS))
a = ap.parse_args()

df = load(a.csv, a.date, a.target)
sel = {k: MODELS[k] for k in a.models.split(",")}
res, preds, nf = rolling_origin(df, a.horizon, a.folds, sel)
res.to_csv("fold_results.csv", index=False)

summ = res.groupby("model")[["MAE", "RMSE", "MAPE%", "sMAPE%", "sec"]].mean().sort_values("MAE")
base = summ.loc["Seasonal Naive", "MAE"] if "Seasonal Naive" in summ.index else None
if base: summ["vs naive"] = ((1 - summ["MAE"] / base) * 100).round(1).astype(str) + "%"
md = [f"Backtest: {nf} folds x {a.horizon}-day horizon (expanding window), {len(df)} daily observations", "",
      summ.round(2).to_markdown()]
open("RESULTS.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))

# plot last fold + error by model
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
tail = df.iloc[-(a.horizon * 4):]; ax[0].plot(tail["ds"], tail["y"], "k", lw=2, label="Actual")
for name, lst in preds.items(): ax[0].plot(lst[-1][0], lst[-1][1], "--", label=name)
ax[0].set_title("Last fold forecast vs actual"); ax[0].legend(); ax[0].tick_params(axis="x", rotation=30)
summ["MAE"].plot.barh(ax=ax[1], color="#4c78a8"); ax[1].invert_yaxis(); ax[1].set_title("Mean MAE across folds (lower = better)")
plt.tight_layout(); plt.savefig("forecast.png", dpi=110); print("saved forecast.png")
