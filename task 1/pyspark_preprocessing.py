
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, avg, to_timestamp, hour, minute, dayofmonth
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import MinMaxScaler, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

INPUT_CSV_PATH = "data/raw/sensor_data.csv"  
OUTPUT_PATH = "data/processed/cleaned_data.parquet"

spark = SparkSession.builder.appName("DataPreprocessing").getOrCreate()

df = spark.read.option("header", True).csv(INPUT_CSV_PATH)


numeric_cols = ["temperature", "humidity"]
for c in numeric_cols:
    df = df.withColumn(c, when(col(c).rlike(r"^\s*$"), None).otherwise(col(c).cast(DoubleType())))


impute_values = {}
for c in numeric_cols:
    mean_val = df.select(avg(col(c))).first()[0]
    if mean_val is None:
        mean_val = 0.0
    impute_values[c] = mean_val

cat_cols = ["status"] if "status" in df.columns else []
for c in cat_cols:
    mode_row = df.groupBy(c).count().orderBy(col("count").desc()).first()
    mode_val = mode_row[0] if mode_row is not None else "unknown"
    impute_values[c] = mode_val

df = df.na.fill(impute_values)


if set(["id", "timestamp"]).issubset(set(df.columns)):
    df = df.dropDuplicates(["id", "timestamp"])
else:
    df = df.dropDuplicates()


if "timestamp" in df.columns:
    df = df.withColumn("ts", to_timestamp(col("timestamp")))
    df = df.withColumn("hour", hour(col("ts"))).withColumn("minute", minute(col("ts"))).withColumn("day", dayofmonth(col("ts")))


features_for_scaling = [c for c in numeric_cols if c in df.columns]
assembler = VectorAssembler(inputCols=features_for_scaling, outputCol="raw_features")
minmax = MinMaxScaler(inputCol="raw_features", outputCol="scaled_features")
standard = StandardScaler(inputCol="raw_features", outputCol="std_features", withStd=True, withMean=False)

pipeline = Pipeline(stages=[assembler, minmax, standard])
model = pipeline.fit(df)
transformed = model.transform(df)

from pyspark.ml.functions import vector_to_array
transformed = transformed.withColumn("scaled_arr", vector_to_array(col("scaled_features")))
for i, fname in enumerate(features_for_scaling):
    transformed = transformed.withColumn(f"{fname}_minmax", col("scaled_arr")[i])


transformed.write.mode("overwrite").parquet(OUTPUT_PATH)

spark.stop()
