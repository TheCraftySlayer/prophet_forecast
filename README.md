# Prophet Forecast Analysis

This project forecasts customer service call volume using [Prophet](https://github.com/facebook/prophet). It merges historical call, visitor and chatbot query data and trains a forecasting model. The script also produces diagnostic charts and exports predictions for the next business days. The model now uses only visitor and query counts plus a single policy indicator as additive regressors to avoid multicollinearity.

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

## Usage

Run the analysis by providing the three input CSV files and an output directory:

```bash
python prophet_analysis.py calls.csv visitors.csv queries.csv output_dir
```

### Optional arguments

 - `--handle-outliers METHOD` – handle detected outliers using `winsorize`, `prediction_replace` or `interpolate`.
- `--use-transformation BOOL` – apply a log transformation to the target before modeling (`true` or `false`).
- `--skip-feature-importance` – skip the feature importance analysis step.
- `--cross-validate` – run full Prophet cross-validation after training.

For example:

```bash
python prophet_analysis.py calls.csv visitors.csv queries.csv prophet_results \
    --handle-outliers winsorize --use-transformation false --cross-validate
```

### Data exclusions

The preprocessing step removes weekends and county holiday closures from the
training set. Any days with zero recorded calls are flagged and treated as
missing values. Call volumes above the 99th percentile are winsorized to
reduce the impact of extreme spikes.

The results, including forecasts and plots, will be saved in the specified output directory.
The exported Excel report now includes predictions for the previous 14 business days
along with a forecast for the next business day. A naive baseline forecast for the
same 14-day window and corresponding MAE, RMSE, and MAPE metrics are also included.
