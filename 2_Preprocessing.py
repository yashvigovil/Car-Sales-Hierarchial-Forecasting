# %%
# load validation sales data from ingestion layer
SOURCE_TABLE = (
    "workspace.gold.car_sales_raw_validated"
)

df = spark.table(SOURCE_TABLE)

display(df.limit(10))

# %%
# keep only business relevant columns reqqiure for forecast
from pyspark.sql.functions import col
forecast_df = df.select(
    col("Date").alias("date"),
    col("Dealer_Region").alias("region"),
    col("Company").alias("company"),
    col("Body_Style").alias("body_style"),
    col("Model").alias("model"),
    col("Color").alias("color")
)

display(forecast_df)

# %%
# convert transcation level data into monthly forecast granualarity each row represnt one car sale
from pyspark.sql.functions import (
    trunc,
    lit
)
forecast_df = (
    forecast_df
    .withColumn(
        "ds",
        trunc("date", "month")
    )
    .withColumn(
        "y",
        lit(1)
    )
)

display(forecast_df)

# %%
# create region level monthly sales hierarchy
from pyspark.sql.functions import (
    concat_ws,
    sum
)
region_df = (
    forecast_df
    .groupBy(
        "ds",
        "region"
    )
    .agg(
        sum("y").alias("y")
    )
    .withColumn(
        "unique_id",
        concat_ws(
            "/",
            lit("Region"),
            col("region")
        )
    )
)
display(region_df)

# %%
# create compnay level monthly sales hierarchy
company_df = (
    forecast_df
    .groupBy(
        "ds",
        "company"
    )
    .agg(
        sum("y").alias("y")
    )
    .withColumn(
        "unique_id",
        concat_ws(
            "/",
            lit("Company"),
            col("company")
        )
    )
)
display(company_df)

# %%
# create body_style level monthly sales hierarchy
body_style_df = (
    forecast_df
    .groupBy(
        "ds",
        "body_style"
    )
    .agg(
        sum("y").alias("y")
    )
    .withColumn(
        "unique_id",
        concat_ws(
            "/",
            lit("BodyStyle"),
            col("body_style")
        )
    )
)

# %%
# create model level monthly sales hierarchy
model_df = (
    forecast_df
    .groupBy(
        "ds",
        "model"
    )
    .agg(
        sum("y").alias("y")
    )
    .withColumn(
        "unique_id",
        concat_ws(
            "/",
            lit("Model"),
            col("model")
        )
    )
)
display(model_df)

# %%
# create color level monthly_sales hierarchy
color_df = (
    forecast_df
    .groupBy(
        "ds",
        "color"
    )
    .agg(
        sum("y").alias("y")
    )
    .withColumn(
        "unique_id",
        concat_ws(
            "/",
            lit("Color"),
            col("color")
        )
    )
)
display(color_df)

# %%
# combine company+region for local level demand
region_company_df = (
    forecast_df
    .groupBy(
        "ds",
        "region",
        "company"
    )
    .agg(
        sum("y").alias("y")
    )
    .withColumn(
        "unique_id",
        concat_ws(
            "/",
            lit("Region"),
            col("region"),
            lit("Company"),
            col("company")
        )
    )
)
display(region_company_df)

# %%
# combine region+compnay+body_style
region_company_body_df = (
    forecast_df
    .groupBy(
        "ds",
        "region",
        "company",
        "body_style"
    )
    .agg(
        sum("y").alias("y")
    )
    .withColumn(
        "unique_id",
        concat_ws(
            "/",
            lit("Region"),
            col("region"),
            lit("Company"),
            col("company"),
            lit("BodyStyle"),
            col("body_style")
        )
    )
)
display(region_company_body_df)

# %%
# combination of region +compnay+body_style+model
region_company_model_df = (
    forecast_df
    .groupBy(
        "ds",
        "region",
        "company",
        "body_style",
        "model"
    )
    .agg(
        sum("y").alias("y")
    )
    .withColumn(
        "unique_id",
        concat_ws(
            "/",
            lit("Region"),
            col("region"),
            lit("Company"),
            col("company"),
            lit("BodyStyle"),
            col("body_style"),
            lit("Model"),
            col("model")
        )
    )
)
display(region_company_model_df)

# %%
# bottom level hierarchy according to granularity
leaf_df = (
    forecast_df
    .groupBy(
        "ds",
        "region",
        "company",
        "body_style",
        "model",
        "color"
    )
    .agg(
        sum("y").alias("y")
    )
    .withColumn(
        "unique_id",
        concat_ws(
            "/",
            lit("Region"),
            col("region"),
            lit("Company"),
            col("company"),
            lit("BodyStyle"),
            col("body_style"),
            lit("Model"),
            col("model"),
            lit("Color"),
            col("color")
        )
    )
)
display(leaf_df)

# %%
hier_df = (
    region_df.select("unique_id","ds","y")
    .union(company_df.select("unique_id","ds","y"))
    .union(body_style_df.select("unique_id","ds","y"))
    .union(model_df.select("unique_id","ds","y"))
    .union(color_df.select("unique_id","ds","y"))
    .union(region_company_df.select("unique_id","ds","y"))
    .union(region_company_body_df.select("unique_id","ds","y"))
    .union(region_company_model_df.select("unique_id","ds","y"))
    .union(leaf_df.select("unique_id","ds","y"))
)
display(hier_df)

# %%
hier_df.select(
    "unique_id"
).distinct().show(
    50,
    truncate=False
)

# %%
hier_df.filter(
    col("unique_id")
    .contains("/Company/")
).show(
    20,
    truncate=False
)

# %%
hier_df.filter(
    col("unique_id")
    .contains("/Color/")
).show(
    20,
    truncate=False
)

# %%
print(
    "Unique Series:",
    hier_df.select(
        "unique_id"
    ).distinct().count()
)

# %%
series_length = (
    hier_df.groupBy(
        "unique_id"
    )
    .count()
)

display(
    series_length.orderBy("count")
)
# many combination are sell only once this creates sparsity becasue model donot learn season and pattern from that.

# %%
# we take the series which appear in 12 months
valid_series = (
    series_length
    .filter(
        col("count") >= 12
    )
    .select("unique_id")
)
display(valid_series)

# %%
 # inner join hier_df
hier_df = (
    hier_df.join(
        valid_series,
        on="unique_id",
        how="inner"
    )
)

# %%
display(hier_df)

# %%
(
    hier_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(
        "workspace.gold.car_sales_hierarchy_filtered"
    )
)

