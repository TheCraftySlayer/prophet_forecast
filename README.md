# Prophet Forecast Analysis

This project forecasts customer service call volume using [Prophet](https://github.com/facebook/prophet). It merges historical call, visitor and chatbot query data and trains a forecasting model. The script also produces diagnostic charts and exports predictions for upcoming business days. In addition to visitor and query counts, the model now includes explicit flags for notice-of-value mail-outs, assessment deadlines, county holidays and a short campaign period. Prophet's built‑in yearly and weekly seasonalities are disabled in favor of these custom regressors.

## Disclaimer

This repository contains only synthetic demonstration data. It ships with stub versions of `pandas`, a minimal `numpy_stub` module and a few other third‑party libraries for unit testing only. Install the real packages when running the forecast for actual analysis. The authors provide this project as-is without any warranty of accuracy or fitness for a particular purpose.

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
without installing the real dependencies. When running the forecasting script
for real analysis you should ensure the actual packages are installed and tell
the script to use them by setting the `USE_REAL_LIBS` environment variable to
`1`:

```bash
set USE_REAL_LIBS=1  # Windows
# or
export USE_REAL_LIBS=1  # Unix
```

This variable is already set in the provided `run_forecast.bat` script.
The batch file accepts an optional output directory as a second argument. If it
does not run when double-clicked, ensure it uses Windows
line endings (CRLF). Some Git tools may check out the repository with Unix
line endings, which can cause `cmd.exe` to ignore the script. You can convert
the file with a tool like `unix2dos` or by opening and resaving it in a
Windows text editor.

## Usage

Before running the scripts ensure the real third-party packages are used by
setting the environment variable `USE_REAL_LIBS` to `1`. This disables the
lightweight stub modules bundled with the repository.

Run the analysis by providing the three input CSV files and an optional output directory:

```bash
set USE_REAL_LIBS=1 && python pipeline.py config.yaml            # Windows
# or
USE_REAL_LIBS=1 python pipeline.py config.yaml                    # Unix
```

The CLI now serves as a thin wrapper around the YAML-driven pipeline. All model
parameters continue to be read from `config.yaml`. The command simply overrides
the input and output paths before calling the pipeline.

If your data resides in a SQLite database you can point the loader to the
appropriate tables using a `path.db:table` notation in `config.yaml`:

```yaml
data:
  calls: mydata.db:calls
  visitors: mydata.db:visits
  queries: mydata.db:queries
```

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
```

This merges the raw files on a business-day index and adds dummy flags for
holidays, notice mail-outs and the May 2025 campaign period.

### Data exclusions

The preprocessing step now creates a continuous daily index. Weekend rows are
inserted with zero call, visit and chatbot counts. Any zero values on weekdays
are flagged and treated as missing so they can be interpolated. Call volumes
above the 99th percentile are winsorized to
reduce the impact of extreme spikes. Dummy variables mark periods for notice
mail-outs, assessment deadlines, a May 2025 campaign and nearby county
holidays. Only a 3‑day moving average of visitor counts and raw chatbot
queries are retained as regressors.

The modeling pipeline applies a `log1p` transform to the target series to
stabilize variance and then back‑transforms predictions to the original scale.

The results, including forecasts and plots, will be saved in the specified output directory.
The exported Excel report (`prophet_call_predictions_v3.xlsx`) now includes
predictions for the previous 14 business days along with a forecast for the next
business day. A seasonal naive baseline forecast using the call volume from the
same weekday in the prior week and the corresponding MAE and RMSE metrics
are also included.

## Model specification

The Prophet model uses additive seasonality with linear growth. Default
hyperparameters are now tuned for a more flexible trend:

- `changepoint_prior_scale=0.4`
- `changepoint_range=0.9`
- `n_changepoints=25`
- `holidays_prior_scale=5`
- `seasonality_prior_scale=0.01`
- `uncertainty_samples=300`
- `regressor_prior_scale=0.05`
- `likelihood=normal`

You can modify these settings in `config.yaml` if desired.

## Cross-validation discipline

The model is evaluated using a rolling origin cross‑validation scheme.
The default initial window spans 365 days with a 30-day horizon and updates every 7 days. A model is accepted only if the mean absolute
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
