# Databricks notebook source
raw_data = spark.read.format("delta").load("/databricks-datasets/nyctaxi-with-zipcodes/subsampled")
display(raw_data)

# COMMAND ----------

raw_data.select('tpep_pickup_datetime').max()

# COMMAND ----------

raw_data.agg({"tpep_pickup_datetime": "max"}).collect()

# COMMAND ----------

raw_data.agg({"tpep_pickup_datetime": "min"}).collect()

# COMMAND ----------


