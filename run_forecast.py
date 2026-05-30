import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp
from hierarchicalforecast.utils import aggregate
import warnings
warnings.filterwarnings("ignore")

print("1. Loading raw dataset...")
df = pd.read_csv("car_sales_extended_raw.csv", low_memory=False)

print("2. Cleaning columns...")
cleaned_columns = [
    col_name.strip()
    .lower()
    .replace(" ", "_")
    .replace("(", "")
    .replace(")", "")
    .replace("$", "")
    .replace("-", "_")
    for col_name in df.columns
]
df.columns = cleaned_columns

# Parse date
df['ds'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True).dt.date

# Create unique_id
df['unique_id'] = (
    df['dealer_region'] + "/" + 
    df['company'] + "/" + 
    df['model'] + "/" + 
    df['color']
)

print("3. Aggregating sales data to daily level...")
forecast_df = df.groupby(['unique_id', 'ds']).size().reset_index(name='y')

# Prepare leaf dataframe
leaf_df = forecast_df.copy()
leaf_df[['region', 'company', 'model', 'color']] = leaf_df['unique_id'].str.split('/', expand=True)
leaf_df = leaf_df[['region', 'company', 'model', 'color', 'ds', 'y']].sort_values(['region', 'company', 'model', 'color', 'ds'])
leaf_df = leaf_df.fillna('ALL')

print("4. Building hierarchy...")
spec = [
    ["region"],
    ["company"],
    ["region", "company"],
    ["region", "company", "model"],
    ["region", "company", "model", "color"]
]

Y_df, S_df, tags = aggregate(df=leaf_df, spec=spec)
print(f"Historical Data Shape: {Y_df.shape}")
print(f"Summation Matrix Shape: {S_df.shape}")

print("5. Running StatsForecast with AutoARIMA...")
sf = StatsForecast(
    models=[AutoARIMA(season_length=12)],
    freq="MS",
    n_jobs=1
)

forecast_reconcile = sf.forecast(df=Y_df, h=12)
print("Forecast generation completed.")

print("6. Performing Hierarchical Reconciliation (Bottom-Up)...")
reconciler = HierarchicalReconciliation(
    reconcilers=[BottomUp()]
)

reconciled_forecast = reconciler.reconcile(
    Y_hat_df=forecast_reconcile,
    S_df=S_df,
    tags=tags
)

# Reset index to ensure unique_id and ds are columns in the CSV
reconciled_forecast = reconciled_forecast.reset_index()

print("7. Saving reconciled forecasts to CSV...")
reconciled_forecast.to_csv("reconciled_forecast.csv", index=False)
print("reconciled_forecast.csv generated successfully.")
