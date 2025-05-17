# Prophet Forecast

This repository contains a Python script (`prophet_analysis.py`) for forecasting call volumes using [Facebook Prophet](https://facebook.github.io/prophet/). Example data for calls, website visitors and chatbot queries are included in `calls.csv`, `visitors.csv` and `queries.csv`.

## Running the analysis on Windows

A helper batch file `run forecast.bat` is provided for convenience. The script changes to the directory where the batch file is located and then runs:

```bat
python prophet_analysis.py calls.csv visitors.csv queries.csv prophet_results --handle-outliers winsorize --use-transformation false
```

Place your data files in the same directory as the batch file and double‑click `run forecast.bat` to execute the pipeline. Results will be written to the `prophet_results` folder.

You can also run the Python script manually with:

```bash
python prophet_analysis.py <calls.csv> <visitors.csv> <queries.csv> [output_dir] [--handle-outliers METHOD] [--use-transformation BOOL] [--skip-feature-importance]
```

See the code comments in `prophet_analysis.py` for more details on the optional arguments.
