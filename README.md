# Prophet Forecast Analysis

This project forecasts customer service call volume using [Prophet](https://github.com/facebook/prophet). It merges historical call, visitor and chatbot query data and trains a forecasting model. The script also produces diagnostic charts and exports predictions for upcoming business days. In addition to visitor and query counts, the model now includes explicit flags for notice-of-value mail-outs, assessment deadlines, county holidays and a short campaign period. Prophet's built‑in weekly seasonality is disabled in favor of a custom component while yearly seasonality is automatically enabled when sufficient history is available.

## Disclaimer

This repository includes real historical CSVs for call volume, chatbot queries and visitor counts. It still ships with stub versions of `pandas`, a minimal `numpy_stub` module and a few other third‑party libraries for unit testing only. Install the real packages when running the forecast for actual analysis. The authors provide this project as-is without any warranty of accuracy or fitness for a particular purpose.

## Requirements

- Python 3.10+ (tested with Python 3.13)
- pandas
- numpy
- matplotlib
- seaborn (optional for some visualizations)
- scikit-learn
- prophet>=1.1.5
- openpyxl
- ruamel.yaml>=0.17.32

Install dependencies with the exact versions pinned in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The pipeline requires `statsmodels` **0.14.2** or newer when using
`pandas` 2.0+. Older releases of `statsmodels` still reference the
removed `pandas.util` module. To resolve a version clash either upgrade
`statsmodels` or pin pandas below 2.0 **but do not mix both approaches**:

```bash
# Option 1: upgrade statsmodels for pandas >= 2.0
python -m pip install --upgrade "statsmodels>=0.14.2"

# Option 2: keep the older statsmodels and install pandas < 2.0
python -m pip install "pandas<2.0"
```

### Stub libraries

For testing purposes the repository contains lightweight stub versions of
`pandas`, a small `numpy_stub` module and a few other libraries. They allow the unit tests to run
without installing the real dependencies. The real packages are used by default.
Set the environment variable `USE_STUB_LIBS` to `1` if you want to force the stub
implementations:

```bash
set USE_STUB_LIBS=1  # Windows
# or
export USE_STUB_LIBS=1  # Unix
```

This variable is already set in the provided `run_forecast.bat` script, which
honors a `PYTHON` environment variable so you can choose a custom interpreter.
If no interpreter is specified, the batch file automatically uses a
`.venv\Scripts\python.exe` next to the repository when available. If it
does not run when double-clicked, ensure it uses Windows
line endings (CRLF). Some Git tools may check out the repository with Unix
line endings, which can cause `cmd.exe` to ignore the script. You can convert
the file with a tool like `unix2dos` or by opening and resaving it in a
Windows text editor.

## Usage

The real third-party packages are used by default. If you need to run the tests
with the lightweight stubs set `USE_STUB_LIBS=1`.

Run the analysis using the YAML configuration:

```bash
python pipeline.py config.yaml            # Windows
# or
python pipeline.py config.yaml                   # Unix
```

Use `--check-baseline-coverage` to verify the naive baseline coverage before
running the model:

```bash
python pipeline.py config.yaml --check-baseline-coverage
```

Alternatively `--baseline-coverage` prints the value and exits with code 1 when
the coverage falls outside the 88–92% range:

```bash
python pipeline.py config.yaml --baseline-coverage
```

The CLI now serves as a thin wrapper around the YAML-driven pipeline. All model
parameters, including input and output paths, are read from `config.yaml`.

If your data resides in a SQLite database you can point the loader to the
appropriate tables using a `path.db:table` notation in `config.yaml`:

```yaml
data:
  calls: mydata.db:calls
  hourly_calls: hourly_call_data.csv  # optional hourly data
  visitors: mydata.db:visits
  queries: mydata.db:queries
```

To model at the hourly level set `model.use_hourly: true` in the YAML and
provide the `hourly_calls` path. The pipeline will forecast the hourly series
and aggregate the results to daily totals. Adjust `model.hourly_periods` to
control how many hours to predict.

The pipeline detects the `.db` extension and automatically queries the database
instead of reading CSV files.

### Data pipeline

To run only the preprocessing step and export a single CSV with engineered
features, use `data_pipeline.py`:

```bash
# use paths from config.yaml
python data_pipeline.py --config config.yaml --out features.csv

# or specify the CSV files directly
python data_pipeline.py calls.csv visitors.csv queries.csv --out features.csv
# optionally include hourly call data to update recent totals
python data_pipeline.py hourly_call_data.csv visitors.csv queries.csv --out features.csv
```

This merges the raw files on a business-day index and adds dummy flags for
holidays, notice mail-outs and the May 2025 campaign period. An accompanying
`assessor_events.csv` lists additional policy and outage dates that can be
used for further feature engineering.

### Hourly forecasting

`hourly_analysis.py` provides a lightweight helper to forecast call volume on an
hourly basis and then sum the predictions to daily totals. It falls back to a
naive mean forecast when the real ``prophet`` package is unavailable so the unit
tests still run with the bundled stubs.

The hourly workflow now records the mean and standard deviation for each
weekday-hour combination. These statistics are written to ``hourly_stats.csv``
whenever the model retrains. Two regressors expose them to Prophet: ``open_flag``
marks business hours while ``mean_hour`` represents the typical call volume for
that hour of the week. A new model is only promoted when it beats the naive mean
benchmark by at least 10 % in MAE or WAPE.

Alternatively, the YAML-driven pipeline can perform the same operation when
`model.use_hourly` is enabled.

```bash
python hourly_analysis.py hourly_call_data.csv --periods 168 --out hourly_forecast.csv
```

The script writes the raw hourly forecast to ``hourly_forecast.csv`` and the
aggregated daily totals to ``daily_forecast.csv``.

Accuracy at the 15‑ and 30‑minute level can be assessed with
``compute_interval_accuracy`` which compares consecutive interval predictions
against the actual counts.

Forecasting assumes the call centre only operates Monday–Friday between
08:00 and 17:00. Any hourly records outside this window are removed prior to
training and evaluation so metrics reflect normal operating hours.

### Data exclusions

The preprocessing step now creates a continuous daily index. Weekend rows are
inserted with zero call, visit and chatbot counts. Any zero values on weekdays
are flagged and treated as missing so they can be interpolated. Zero-call days
are evaluated separately using a hurdle-style metric so they do not distort
percentage errors. Call volumes
above the 99.5th percentile are winsorized to
reduce the impact of extreme spikes. Dummy variables mark periods for notice
mail-outs, assessment deadlines, a May 2025 campaign and nearby county
holidays. Only a 3‑day moving average of visitor counts and raw chatbot
queries are retained as regressors. Standardised 7‑day lags of visitor and query counts capture over 65% of the explainable variance one week ahead. Lagged call counts at 1‑ and 7‑day intervals are also used to mitigate autocorrelation.
When residual autocorrelation persists, an autoregressive adjustment using lags 1 and 7 is applied to the predictions.

The modeling pipeline applies either a logarithmic or Box‑Cox transform to the
target series (controlled by `model.transform` in `config.yaml`). By default a
`log1p` transform is used and predictions are back‑transformed to the original
scale.

The results, including forecasts and plots, will be saved in the specified output directory.
The exported CSV file (`prophet_call_predictions_<hash>.csv`) now includes
predictions for the previous 14 business days along with a forecast for the next
business day. A seasonal naive baseline forecast now uses call volume from the
same hour and weekday one week earlier, aggregated to daily totals. The
corresponding MAE, RMSE and Poisson deviance metrics are also included. The
report additionally lists the predicted call volume for the upcoming business
day in a dedicated sheet.

Each run logs the training window, any parameter overrides and a SHA1 hash of the
serialized model so forecasts can be exactly reproduced.

## Model specification

The Prophet model uses additive seasonality with linear growth. Default
hyperparameters are now tuned for a more flexible trend:

- `changepoint_prior_scale` searched from 0.05–0.5
- `changepoint_range=0.9`
- `n_changepoints` searched between 10 and 40
- `holidays_prior_scale=5`
- `seasonality_prior_scale=0.01`
- `uncertainty_samples=300`
- `regressor_prior_scale=0.05`
- `likelihood=auto` (automatically selects Poisson or negative-binomial)
- `yearly_seasonality=auto`
- `weekly_seasonality=auto` or a custom weekly component with `fourier_order=5` when explicitly disabled
- `capacity` sets the logistic growth cap (defaults to 110% of training max)

Hyperparameter tuning relies on rolling cross‑validation. The grid explores
 changepoint scales from 0.05–0.5 and between 10 and 40 changepoints alongside
 the seasonality and holiday priors. This helps avoid under‑ or over‑fitting and
 targets an MAE no greater than 62 with WAPE under 32%.

You can modify these settings in `config.yaml` if desired. Event windows such as
campaign dates, policy start and any explicit changepoints also live in the
configuration file. Set `enable_mcmc: true` only when you purposely want
Bayesian sampling, otherwise any non-zero `mcmc_samples` value is ignored.

## Cross-validation discipline

The model is evaluated using a rolling origin cross‑validation scheme.
The default initial window spans 180 days with a 14-day horizon and updates every 30 days. A model is accepted only if the mean absolute
error stays below 15% of the average call volume.

### Scaling details

The target series ``call_count`` is standardized prior to model fitting. Each
run saves the fitted ``StandardScaler`` as ``call_scaler.pkl`` in the output
directory. The pickle stores the mean and standard deviation used for the
transformation. During evaluation and when exporting forecasts the predictions
are inverse transformed so that metrics reflect the original call counts.
When reusing the scaler ensure the underlying data distribution matches the
stored parameters.

### Windows compatibility

Prophet's diagnostics spawn multiple worker processes. On Windows the
default Tkinter-backed Matplotlib GUI conflicts with this and can raise
``Operation not permitted`` errors from CmdStan. The analysis script now
forces a headless Matplotlib backend and runs cross‑validation using
threads to avoid the issue.

## Running the tests

A comprehensive suite of unit tests exercises the data pipeline and model components. After installing the required dependencies you can execute the tests with:

```bash
pytest
```

The tests run against the real third-party packages by default. Set `USE_STUB_LIBS=1` to use the lightweight stub modules shipped in this repository instead.

### Makefile helpers

The repository includes a small `Makefile` exposing common tasks:

```bash
make features  # preprocess the raw data
make model     # train the Prophet model
make metrics   # echo where metrics are exported
```

## Forecast diagnostics

Recent evaluations uncovered structural drift in the hourly series:

- **Error growth** – MAE nearly doubled between February and May while RMSE
  climbed even faster, pointing to heavy-tailed risk.
- **Bias instability** – alternating bias signs suggest weekly seasonality no
  longer matches demand.
- **Volatility surge April–May** – spikes in RMSE indicate unmodelled shocks
  such as policy deadlines or outreach campaigns.
- **Scale-dependent error** – MAPE stays above 60%, meaning quiet periods are
  disproportionately affected.

To mitigate these issues the pipeline now performs residual and bias
monitoring. A retrain is triggered whenever residuals exceed twice the rolling
14‑day MAE **or** when absolute bias surpasses ±5 calls for three consecutive
days. Adding event-based regressors and more frequent retraining keeps the
model in sync with shifting patterns, while explicit variance checks allow the
forecast to react quickly to demand spikes.

