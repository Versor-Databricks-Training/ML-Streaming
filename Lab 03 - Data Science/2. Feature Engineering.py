# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Feature Engineering
# MAGIC 
# MAGIC We will now make use of our learnings from our exploratory analysis and build our feature engineering pipeline. 
# MAGIC 
# MAGIC We often refer to the structured inputs as a collection of features. From these features we may apply non-linear transformations and produce new **derived** features. 
# MAGIC 
# MAGIC Typically, these derived features are informed by input from subject matter experts.
# MAGIC 
# MAGIC In principle, a sufficiently expressive machine learning model would have the ability to learn the non-linear maps, if the derived feature is indeed important. However this may require more training data than is available and so the engineering of derived features can lighten the workload of the model.  
# MAGIC 
# MAGIC If the features turn out to not be interesting, then the model should de-emphasize them. In this manner, feature engineering is an excellent way to test commonly held beliefs from subject experts.
# MAGIC 
# MAGIC Carefully curated features are valuable and must be shared amongst collegues. For this reason, Databricks has developed a specific UI for derived features.

# COMMAND ----------

# MAGIC %md
# MAGIC # Derived Features
# MAGIC 
# MAGIC There are two methods to compute derived features:  
# MAGIC <br> 
# MAGIC 1. **Precompute**
# MAGIC 
# MAGIC When features are precomputed, they require a **feature store**, where they are stored and from where they will be retrieved at inference.  Typically these precomputed features are aggregations over multiple samples, so that the feature cannot even in principle be computed when performing inference on a single sample. 
# MAGIC 
# MAGIC One can also pre-compute derived features on a single sample; indeed this is similar to the familiar numerical trick of reducing compute time by approximating a function with a lookup table.  
# MAGIC <br> 
# MAGIC 2. **Compute at time of inference**
# MAGIC 
# MAGIC Features which are computed at time of inference must only depend on a single sample. Nonetheless they could still be a non-linear map and possibly be compute intensive. When computing features at the time of inference, this is typically done in a **pipeline**, which we will study in the next notebook.

# COMMAND ----------

# MAGIC %run "./Utils/Fetch_User_Metadata"

# COMMAND ----------

# DBTITLE 1,Retrieve the experiment data from our database
spark.sql(f"USE {DATABASE_NAME}")
data = spark.table("phytochemicals_quality")

# COMMAND ----------

# DBTITLE 1,We can use our familiar pandas commands for data science (without sacrificing scalability)
import pyspark.pandas as ps
raw_data = data.to_pandas_on_spark()

# COMMAND ----------

# MAGIC %md
# MAGIC ## We can create a new feature from the pH value
# MAGIC From chemistry, we know that pH approximates the concentration of hydrogen ions in a solution. We are going to use this information to include a new (potentially predictive) feature into our model: 
# MAGIC 
# MAGIC $$\\text{pH} = - \\text{log}_{10} ( h_{\\text{concentration}} )$$
# MAGIC $$ \Rightarrow h_{\\text{concentration}} = 10^{-\\text{pH}} $$

# COMMAND ----------

raw_data = raw_data.assign(h_concentration=lambda x: 1/(10**x["pH"]))

# COMMAND ----------

# DBTITLE 1,We now look at the distribution of our newly calculated feature - looks good!
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context("paper", font_scale=1.8)
sns.displot(raw_data["h_concentration"].to_numpy())
plt.ylabel("Count")
plt.xlabel("hydrogen concentration (moles)")
plt.show()

# COMMAND ----------

# DBTITLE 1,Our chemists also tell us that the ratio of acidity to sugar may be a useful predictor of quality
raw_data = raw_data.assign(acidity_ratio=lambda x: x["citric_acid"]/x["residual_sugar"])
sns.displot(raw_data["acidity_ratio"].to_numpy())
plt.ylabel("Count")
plt.xlabel("Acidity ratio (no units)")
plt.show()

# COMMAND ----------

# DBTITLE 1,This distribution is quite skewed so we apply a log transformation - looks much better!
import numpy as np

raw_data = raw_data.assign(acidity_ratio=lambda x: np.log(x["citric_acid"]/x["residual_sugar"]))
sns.displot(raw_data["acidity_ratio"].to_numpy())

plt.ylabel("Count")
plt.xlabel("Acidity ratio (no units)")
plt.show()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC # Register Features into the Feature Store 
# MAGIC 
# MAGIC We now register our features into the feature store so others in APJuice can reuse our features for other experiments! The feature store will also make inference easier as the Delta table will record our transformations and reapply these during inference. This applies to both batch and streaming inference. Orange (🍊) you glad you chose Delta!
# MAGIC 
# MAGIC A centralised feature store also allows for discoverability and reusability of our feature accross our organization, increasing team efficiency of data scientists. The feature store can bring traceability and governance in your deployments, knowing which model is dependent of which set of features.
# MAGIC 
# MAGIC <!-- 
# MAGIC <img src="https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/mlops-end2end-flow-feature-store.png" style="float:right" width="500" />
# MAGIC  -->
# MAGIC Once our features are ready, we'll save them in Databricks Feature Store. Under the hood, features store are backed by a Delta Lake table.
# MAGIC 
# MAGIC 
# MAGIC <div style="text-align:bottom">
# MAGIC   <img src="https://ajmal-field-demo.s3.ap-southeast-2.amazonaws.com/apj-sa-bootcamp/feature_store.png" width="1100px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Features Overview

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC A feature store functions as a lookup table requires a primary key. This must be chosen judiciously as this will be used  
# MAGIC to join the feature store with unseen data at inference time. So the unseen data must have a primary key  
# MAGIC which is in the training data.  
# MAGIC 
# MAGIC In the current example, we are bypassing this concern since train/valid/test are all known  
# MAGIC from the outset and their features are in the feature store.

# COMMAND ----------

# DBTITLE 0,We now register our features into the feature store
from databricks import feature_store

fs = feature_store.FeatureStoreClient()

fs.create_table(
  name=f"{DATABASE_NAME}.features_oj_prediction_experiment",
  primary_keys=["customer_id"],
  df=raw_data.to_spark(),
  description="""
  Features for predicting the quality of an orange. 
  Additionally, I have a calculated column called acidity_ratio=log(citric_acid/residual sugar) as well as calculating the hydrogen concentration.
  """
)

displayHTML("""
  <h3>Check out the <a href="/#feature-store/{}.features_oj_prediction_experiment">feature store</a> to see where our features are stored.</h3>
""".format(DATABASE_NAME))

# COMMAND ----------


