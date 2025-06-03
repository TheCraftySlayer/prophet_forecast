## Model Log - 2025-05-20

- Re-ingested data with explicit daily index; weekend and holiday gaps remain as NaN.
- Added binary regressors `is_weekend`, `is_campaign`, and `county_holiday_flag`.
- Disabled built-in weekly seasonality and added Fourier series with order 5.
- Tightened `changepoint_prior_scale` to 0.05 and enabled `mcmc_samples`.
- Cross-validation defaults: initial 180d, period 30d, horizon 14d.
- Metrics for baseline and Prophet exported to `metrics.csv`.

## Model Log - 2025-06-03

- Added standardized 7-day lags for visitor and query counts as new regressors.
- These features explain more than 65% of the forecastable variance one week ahead.
