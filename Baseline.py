# %%
%pip install statsforecast
%pip install hierarchicalforecast
%pip install utilsforecast

# %%
# import all the requirements
import pandas as pd
import numpy as np

from statsforecast import StatsForecast

from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    SeasonalNaive
)

from hierarchicalforecast.core import (
    HierarchicalReconciliation
)

from hierarchicalforecast.methods import (
    BottomUp,
    MinTrace
)

from pyspark.sql.functions import col

# %%
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("/Volumes/workspace/default/ml-volume/car_sales_extended_raw.csv")

display(df.limit(10))

# %%
df.printSchema()
display(df)

# %%
# Clean column names
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

df = df.toDF(*cleaned_columns)

# Verify new column names
print(df.columns)

# %%
df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    "workspace.gold.car_sales_hierarchy_filtered"
)

# %%
df = spark.table(
    "workspace.gold.car_sales_hierarchy_filtered"
)

display(df.limit(10))

# %%
from pyspark.sql.functions import concat_ws, col

df = df.withColumn(
    "unique_id",
    concat_ws(
        "/",
        col("dealer_region"),
        col("company"),
        col("model"),
        col("color")
    )
)

display(df.select("unique_id").limit(10))

# %%
from pyspark.sql.functions import to_date, count

forecast_df = (
    df.withColumn("ds", to_date(col("date")))
      .groupBy("unique_id", "ds")
      .agg(count("*").alias("y"))
)

display(forecast_df.limit(10))

# %%
# leaf node bottom level
from pyspark.sql.functions import col
leaf_df = df.filter(
    col("unique_id")
    .contains("/Color/")
)

display(leaf_df.limit(10))

# %%
# Nixtla libraries are build for pandas so we convert it to pandas from spark.
pdf = forecast_df.toPandas()

pdf.head()

# %%
print(pdf.columns)

pdf.info()

# %%
# sort it before forecasting.
pdf = pdf.sort_values(
    ["unique_id", "ds"]
)

pdf.head()

# %%
# splitng the dataset into test train and validation
# start three year for training 1 year for validation and 1 year for testing
train_end = pd.to_datetime(
    "2025-01-01"
).date()

validation_end = pd.to_datetime(
    "2026-01-01"
).date()
train_df = pdf[
    pdf["ds"] < train_end
]
validation_df = pdf[
    (
        pdf["ds"] >= train_end
    )
    &
    (
        pdf["ds"] < validation_end
    )
]
test_df = pdf[
    pdf["ds"] >= validation_end
]
print(
    "Train Shape:",
    train_df.shape
)
print(
    "Validation Shape:",
    validation_df.shape
)
print(
    "Test Shape:",
    test_df.shape
)

# %%
sf = StatsForecast(
    models=[
        AutoARIMA(
            season_length=12
        )
    ],

    freq="MS",
    n_jobs=-1
)

# %%
validation_forecast = (
    sf.forecast(
        df=train_df,
        h=12
    )
)

validation_forecast.head()

# %%
validation_forecast["ds"] = pd.to_datetime(
    validation_forecast["ds"]
)

validation_df["ds"] = pd.to_datetime(
    validation_df["ds"]
)

# %%
# compare the result with validation data
validation_comparison = (
    validation_forecast.merge(
        validation_df,
        on=["unique_id", "ds"],
        how="inner"
    )
)

validation_comparison.head()

# %%
# measure the accuracy of auti arima
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

import numpy as np


validation_mae = mean_absolute_error(
    validation_comparison["y"],
    validation_comparison["AutoARIMA"]
)

validation_rmse = np.sqrt(
    mean_squared_error(
        validation_comparison["y"],
        validation_comparison["AutoARIMA"]
    )
)

print(
    "Validation MAE:",
    round(validation_mae, 2)
)

print(
    "Validation RMSE:",
    round(validation_rmse, 2)
)

# %%
# combine train and validation for the retraining so that it has more updated information
train_validation_df = pd.concat(
    [
        train_df,
        validation_df
    ],
    ignore_index=True
)

# %%
print(
    "Combined Shape:",
    train_validation_df.shape
)

train_validation_df.head()

# %%
retrain_sf = StatsForecast(
    models=[
        AutoARIMA(
            season_length=12
        )
    ],

    freq="MS",
    n_jobs=-1
)

# %%
test_forecast = (
    retrain_sf.forecast(
        df=train_validation_df,
        h=12
    )
)

display(test_forecast.head())

# %%
test_forecast["ds"] = pd.to_datetime(
    test_forecast["ds"]
)

test_df["ds"] = pd.to_datetime(
    test_df["ds"]
)

# %%
# merge with the test data
test_evaluation = (
    test_forecast.merge(
        test_df,
        on=["unique_id", "ds"],
        how="inner"
    )
)

test_evaluation.head()

# %%
test_mae = mean_absolute_error(
    test_evaluation["y"],
    test_evaluation["AutoARIMA"]
)

test_rmse = np.sqrt(
    mean_squared_error(
        test_evaluation["y"],
        test_evaluation["AutoARIMA"]
    )
)

print(
    "Test MAE:",
    round(test_mae, 2)
)

print(
    "Test RMSE:",
    round(test_rmse, 2)
)
# means there is no overfitting.

# %%
from hierarchicalforecast.core import (
    HierarchicalReconciliation
)

from hierarchicalforecast.methods import (
    MinTrace
)

from hierarchicalforecast.utils import (
    aggregate
)

# %%
# retrain the model on the whole data so it has latest data.
final_sf = StatsForecast(
    models=[
        AutoARIMA(
            season_length=12
        )
    ],

    freq="MS",
    n_jobs=-1
)

final_forecast = (
    final_sf.forecast(
        df=pdf,
        h=12
    )
)

# %%
print(type(final_forecast))

# %%
# Purpose:
# Define hierarchy
# combinations.


spec = [
    ["region"],
    ["company"],
    ["region", "company"],
    ["region", "company", "model"],
    ["region", "company", "model", "color"]
]

print(spec)

# %% [markdown]
# ## NIXITILA RECONCILATION PHASE

# %%
hierarchy_pdf = (
    df.toPandas()
)

hierarchy_pdf.head()

# %%
# Purpose:
# Keep only complete
# leaf-node hierarchy.


leaf_df = (
    hierarchy_pdf[
        hierarchy_pdf[
            "unique_id"
        ].str.contains(
            "Region"
        )
        &
        hierarchy_pdf[
            "unique_id"
        ].str.contains(
            "Company"
        )
        &
        hierarchy_pdf[
            "unique_id"
        ].str.contains(
            "BodyStyle"
        )
        &
        hierarchy_pdf[
            "unique_id"
        ].str.contains(
            "Model"
        )
        &
        hierarchy_pdf[
            "unique_id"
        ].str.contains(
            "Color"
        )
    ]
)

leaf_df.head()

# %%
# Purpose:
# Extract hierarchy columns
# from unique_id.


def extract_level(uid, level):

    parts = uid.split("/")

    if level in parts:

        idx = parts.index(level)

        if idx + 1 < len(parts):

            return parts[idx + 1]

    return None


# region
leaf_df["region"] = (
    leaf_df["unique_id"]
    .apply(
        lambda x:
        extract_level(
            x,
            "Region"
        )
    )
)

# company
leaf_df["company"] = (
    leaf_df["unique_id"]
    .apply(
        lambda x:
        extract_level(
            x,
            "Company"
        )
    )
)

# body style
leaf_df["body_style"] = (
    leaf_df["unique_id"]
    .apply(
        lambda x:
        extract_level(
            x,
            "BodyStyle"
        )
    )
)

# model
leaf_df["model"] = (
    leaf_df["unique_id"]
    .apply(
        lambda x:
        extract_level(
            x,
            "Model"
        )
    )
)

# color
leaf_df["color"] = (
    leaf_df["unique_id"]
    .apply(
        lambda x:
        extract_level(
            x,
            "Color"
        )
    )
)

leaf_df.head()

# %%
from pyspark.sql.functions import (
    col,
    concat_ws,
    to_date,
    count
)

# Create hierarchy id
df = df.withColumn(
    "unique_id",
    concat_ws(
        "/",
        col("dealer_region"),
        col("company"),
        col("model"),
        col("color")
    )
)

# Create forecasting dataframe
forecast_df = (
    df.withColumn("ds", to_date(col("date")))
      .groupBy("unique_id", "ds")
      .agg(count("*").alias("y"))
)

# %%
leaf_df = forecast_df.toPandas()
print(leaf_df.columns)

# %%
# Purpose:
# Keep only required
# hierarchy columns.

leaf_df[
    ["region", "company", "model", "color"]
] = leaf_df["unique_id"].str.split(
    "/",
    expand=True
)

leaf_df.head()

leaf_df = leaf_df[
    [
        "region",
        "company",
        "model",
        "color",
        "ds",
        "y"
    ]
].sort_values(
    ["region", "company", "model", "color", "ds"]
)

leaf_df.head()

# %%
leaf_df = (
    leaf_df.fillna(
        "ALL"
    )
)

leaf_df.head()

# %%

# Purpose:
# Create hierarchy matrix
# for Bottom-Up.


from hierarchicalforecast.utils import (
    aggregate
)

Y_df, S_df, tags = aggregate(
    df=leaf_df,
    spec=spec
)

print(
    "Historical Data Shape:",
    Y_df.shape
)

print(
    "Summation Matrix Shape:",
    S_df.shape
)

print(
    "Hierarchy Levels:"
)

print(
    tags.keys()
)

# %%
import mlflow
import cloudpickle
import os

# Set your experiment tracking location
mlflow.set_experiment("/Shared/Car_Sales_Hierarchical_Forecasting")

with mlflow.start_run(run_name="AutoARIMA_BottomUp_Run"):
    
    # Log the parameters so you know what settings you used later
    mlflow.log_param("forecast_horizon", 12)
    mlflow.log_param("model", "AutoARIMA")
    mlflow.log_param("reconciler", "BottomUp")

    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA
    
    hierarchy_sf = StatsForecast(
        models=[AutoARIMA(season_length=12)],
        freq="MS",
        n_jobs=-1
    )
    
    forecast_reconcile = hierarchy_sf.forecast(df=Y_df, h=12)

    from hierarchicalforecast.core import HierarchicalReconciliation
    from hierarchicalforecast.methods import BottomUp
    
    reconciler = HierarchicalReconciliation(
        reconcilers=[BottomUp()]
    )
    
    reconciled_forecast = reconciler.reconcile(
        Y_hat_df=forecast_reconcile,
        S_df=S_df,
        tags=tags
    )
    
    # Save the dataframe
    forecast_output_path = "reconciled_forecast.csv"
    reconciled_forecast.to_csv(forecast_output_path, index=False)
    mlflow.log_artifact(forecast_output_path, artifact_path="forecast_data")
    
    # Save the models for deployment
    with open("hierarchical_model.pkl", "wb") as f:
        cloudpickle.dump((hierarchy_sf, reconciler), f)
    mlflow.log_artifact("hierarchical_model.pkl", artifact_path="models")

# %%
reconciled_forecast[
    "unique_id"
].drop_duplicates().sample(
    30
)

# %%
(
    spark.createDataFrame(
        reconciled_forecast
    )
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(
        "workspace.gold.car_sales_final_forecast"
    )
)

