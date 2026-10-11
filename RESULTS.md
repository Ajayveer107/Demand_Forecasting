Backtest: 6 folds x 14-day horizon (expanding window), 1461 daily observations

| model          |   MAE |   RMSE |   MAPE% |   sMAPE% |   sec | vs naive   |
|:---------------|------:|-------:|--------:|---------:|------:|:-----------|
| Prophet        | 27.19 |  33.01 |    1.55 |     1.55 |  0.92 | 32.7%      |
| XGBoost        | 32.45 |  40.52 |    1.86 |     1.85 |  1.01 | 19.7%      |
| Seasonal Naive | 40.4  |  48.65 |    2.3  |     2.28 |  0    | 0.0%       |
