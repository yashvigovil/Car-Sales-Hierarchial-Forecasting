# %%
df = spark.table(
    "workspace.gold.car_sales_hierarchy_filtered"
)

display(df)

# %%
df.printSchema()

# %%
print(df.columns)

# %%
# basic dataset statistics
total_rows = df.count()

print(f"Total Rows: {total_rows:,}")

print(
    f"Total Columns: {len(df.columns)}"
)

# %%
from pyspark.sql.functions import (
    min,
    max
)

date_summary = df.select(
    min("Date").alias("min_date"),
    max("Date").alias("max_date")
)

display(date_summary)

# %%
# no missing mon th in the data also it gives  o of car sold in each month
from pyspark.sql.functions import (
    trunc,
    count
)
monthly_counts = (
    df.withColumn(
        "month",
        trunc("Date", "month")
    )
    .groupBy("month")
    .agg(
        count("*").alias(
            "transaction_count"
        )
    )
    .orderBy("month")
)

display(monthly_counts)

# %%
# critical column has no null values.
from pyspark.sql.functions import (
    col,
    count,
    when,
    round
)

total_rows = df.count()

null_check = df.select([
    count(
        when(col(c).isNull(), c)
    ).alias(c)
    for c in df.columns
])

display(null_check)

# %%
# standarize the column names.
clean_columns = [
    col_name.strip()
    .replace(" ", "_")
    .replace("(", "")
    .replace(")", "")
    .replace("$", "")
    .replace("-", "_")
    for col_name in df.columns
]

df = df.toDF(*clean_columns)

print(df.columns)

# %%
(
    df.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(
        "workspace.gold.car_sales_raw_validated"
    )
)

