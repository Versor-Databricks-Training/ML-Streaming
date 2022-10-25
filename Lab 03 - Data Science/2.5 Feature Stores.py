# Databricks notebook source
# MAGIC %run "./Utils/Fetch_User_Metadata"

# COMMAND ----------

spark.sql(f"USE {DATABASE_NAME}")

# COMMAND ----------

import numpy as np
import pandas as pd


N = 5 
df = pd.DataFrame({'a': range(N), 'b': np.random.random(N), 'c': np.random.random(N), 'label': np.random.random(N)})
data = spark.createDataFrame(df)

# COMMAND ----------

import pyspark.sql.functions as F

def compute_derived_features(data):
  data = data.withColumn('d', F.col('b')/F.col('c'))
  return data

data_features = compute_derived_features(data)

display(data_features)

# COMMAND ----------

from databricks.feature_store import feature_table
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

# create an empty features table
customer_feature_table = fs.create_table(
  name=f'{DATABASE_NAME}.features_test',
  primary_keys='a',
  schema=data_features.schema,
  description='Test features'
)

fs.write_table(
  name=f'{DATABASE_NAME}.features_test',
  df = data_features,
  mode = 'overwrite'
)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC DROP TABLE IF EXISTS features_test;
# MAGIC 
# MAGIC CREATE TABLE features_test
# MAGIC USING DELTA
# MAGIC AS
# MAGIC SELECT * FROM features_data;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * from features_test;

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC alter table features_test
# MAGIC add column c float;

# COMMAND ----------

# MAGIC %sql 
# MAGIC alter table features_test
# MAGIC set c = a

# COMMAND ----------


