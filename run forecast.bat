cd "C:\Users\anewe\OneDrive - Bernalillo County\Customer Data\Python Project"

cmd /k python prophet_analysis.py calls.csv visitors.csv queries.csv prophet_results --handle-outliers winsorize --use-transformation false

