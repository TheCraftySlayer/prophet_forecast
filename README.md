# Prophet Forecast Analysis

This project forecasts customer service call volume using [Prophet](https://github.com/facebook/prophet). It merges historical call, visitor and chatbot query data and trains a forecasting model. The script also produces diagnostic charts and exports predictions for the next business days. In addition to visitor and query counts, the model now includes explicit flags for notice-of-value mail-outs, assessment deadlines and federal holidays while disabling Prophet's built-in yearly seasonality.

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

Install dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn \
    prophet>=1.1.5 openpyxl ruamel.yaml>=0.17.32
```

### Stub libraries

For testing purposes the repository contains lightweight stub versions of
`pandas`, `numpy` and a few other libraries. They allow the unit tests to run
without installing the real dependencies. When running the forecasting script
for real analysis you should ensure the actual packages are installed and tell
the script to use them by setting the `USE_REAL_LIBS` environment variable to
`1`:

```bash
set USE_REAL_LIBS=1  # Windows
# or
export USE_REAL_LIBS=1  # Unix
```

This variable is already set in the provided `run forecast.bat` script.
If the batch file does not run when double-clicked, ensure it uses Windows
line endings (CRLF). Some Git tools may check out the repository with Unix
line endings, which can cause `cmd.exe` to ignore the script. You can convert
the file with a tool like `unix2dos` or by opening and resaving it in a
Windows text editor.

## Usage

Run the analysis by providing the three input CSV files and an output directory:

```bash
python prophet_analysis.py calls.csv visitors.csv queries.csv output_dir
```

### Optional arguments

- `--handle-outliers METHOD` – handle detected outliers using `winsorize`, `prediction_replace` or `interpolate`.
- `--use-transformation BOOL` – apply a `log1p` transformation to the target before modeling and back‑transform predictions (`true` or `false`).
- `--skip-feature-importance` – skip the feature importance analysis step.
- `--cross-validate` – run full Prophet cross-validation after training.
- `--likelihood` – choose the likelihood for the underlying Stan model (`normal`, `poisson`, or `neg_binomial`).

For example:

```bash
python prophet_analysis.py calls.csv visitors.csv queries.csv prophet_results \
    --handle-outliers winsorize --use-transformation false --cross-validate \
    --likelihood poisson
```

### Data exclusions

The preprocessing step removes weekends and county holiday closures from the
training set. Any days with zero recorded calls are flagged and treated as
missing values. Call volumes above the 99th percentile are winsorized to
reduce the impact of extreme spikes. Additional dummy variables mark periods for
notice mail-outs, assessment deadlines and nearby federal holidays. Weekly
patterns are modeled using Prophet's built-in seasonality instead of manual
weekday columns.

The modeling pipeline applies a `log1p` transform to the target series to
stabilize variance and then back‑transforms predictions to the original scale.

The results, including forecasts and plots, will be saved in the specified output directory.
The exported Excel report now includes predictions for the previous 14 business days
along with a forecast for the next business day. A naive baseline forecast for the
same 14-day window and corresponding MAE, RMSE, and MAPE metrics are also included.

## Model specification

The Prophet model uses additive seasonality with linear growth. Default
hyperparameters are:

- `changepoint_prior_scale=0.05`
- `holidays_prior_scale=5`
- `regressor_prior_scale=0.05`
- `likelihood=normal`

You can modify these settings in `config.yaml` if desired.

## Cross-validation discipline

The model is evaluated using a rolling origin cross‑validation scheme.
By default the initial training window spans 270 days with a 30‑day
horizon and evaluation period of 14 days. A model is accepted only if the
mean absolute percentage error (MAPE) stays below 20% and the mean
absolute error (MAE) is under 60 calls.
