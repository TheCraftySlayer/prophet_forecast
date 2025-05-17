# Prophet Forecast Analysis

This project forecasts customer service call volume using [Prophet](https://github.com/facebook/prophet). It merges historical call, visitor and chatbot query data and trains a forecasting model. The script also produces diagnostic charts and exports predictions for the next business days.

## Requirements

- Python 3.10+
- pandas
- numpy
- matplotlib
- seaborn (optional for some visualizations)
- scikit-learn
- prophet
- openpyxl

Install dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn prophet openpyxl
```

## Usage

Run the analysis by providing the three input CSV files and an output directory:

```bash
python prophet_analysis.py calls.csv visitors.csv queries.csv output_dir
```

### Optional arguments

- `--handle-outliers METHOD` – handle detected outliers using `winsorize`, `median_replace` or `interpolate`.
- `--use-transformation BOOL` – apply a log transformation to the target before modeling (`true` or `false`).
- `--skip-feature-importance` – skip the feature importance analysis step.
- `--cross-validate` – run full Prophet cross-validation after training.

For example:

```bash
python prophet_analysis.py calls.csv visitors.csv queries.csv prophet_results \
    --handle-outliers winsorize --use-transformation false --cross-validate
```

The results, including forecasts and plots, will be saved in the specified output directory.
The exported Excel report now includes predictions for the previous 14 business days
along with a forecast for the next business day.
