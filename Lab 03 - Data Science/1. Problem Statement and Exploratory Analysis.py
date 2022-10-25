# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC # Machine Learning Introduction on Databricks
# MAGIC 
# MAGIC We start by understanding the business problem before translating to a technical problem. 
# MAGIC 
# MAGIC ## The Business Problem
# MAGIC APJuice is really interested in having the best quality ingredients for their juices. Their most popular juice flavours are Oranges. You might be thinking, "Simple! How different could oranges really be?" Well, actually, there's over 20 varieties of oranges and even more flavour profiles. The key indicators for flavour profiles of an orange are: Level of **acidity**, amount of **enzymes**, **citric acid** concentration, **sugar content**, **chlorides**, the aroma (**Octyl Acetate**), and amount of **sulfur dioxide**.
# MAGIC 
# MAGIC Clearly the flavour profile of an orange is quite complex. Additionally, easy of these variables that determine the taste have **differing marginal cost**. For example, increasing the amount of Octyl Acetate is **more expensive** than the amount of sugar. 
# MAGIC 
# MAGIC 
# MAGIC <div style="text-align:center">
# MAGIC   <img src="https://ajmal-field-demo.s3.ap-southeast-2.amazonaws.com/apj-sa-bootcamp/orange_classification.png" width="1000px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC APJuice are quite scientific and follow the Popperian view of science. Additionally, they are willing to accept that if a model can be derived that can model this relationship then the hypothesis has been proven true.
# MAGIC 
# MAGIC > **Hypothesis statement:** do the chemical properties influence the taste of an orange? If so, what is the best combination of chemical properties (financially) such that the quality is high but the cost is low?
# MAGIC 
# MAGIC As a starting point, APJuice collected some data from customers and what they thought of the quality of some oranges. We will test the hypothesis by training a machine learning model to predict quality scores from respondents. Let's start with some exploratory data analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Workflow Outline

# COMMAND ----------

# MAGIC %md
# MAGIC 1. **Exploratory Data Analysis**
# MAGIC     * all good data scientists start by graphing the data
# MAGIC     * correlations between variables can be interesting
# MAGIC 
# MAGIC 2. **Feature Generation**
# MAGIC     * New Features would be non-linear relationships between existing features.  
# MAGIC      Usually representing expert business knowledge.
# MAGIC     * Databricks has a convenient UI to search and share features at the workspace level.
# MAGIC     * Features will have to be created for all "out-of-box" data.
# MAGIC 
# MAGIC 3. **Pipeline**
# MAGIC     * Impute missing data
# MAGIC     * Encode categorial data
# MAGIC     * Rescale numerical data
# MAGIC     * Predict target variable
# MAGIC     * The pipeline must be prepared on the training data, then persisted.
# MAGIC     * The pipeline will then have to be run on all "out-of-box" data
# MAGIC    
# MAGIC 4. **Cross Validation**
# MAGIC     * A predictive model has to be evaluated on a (at least one) test set.
# MAGIC     * Choice of evaluation metric is cruial and should be developed  
# MAGIC       in a collaboration between data scientists and business/product owners.
# MAGIC 
# MAGIC 5. **Deployment**
# MAGIC     * The model should be made available either internally or externally (or both)
# MAGIC     * In many cases, REST APIs are particularly useful for deployment.  
# MAGIC     * One must pay attention to latency of the prediction

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Exploratory Data Analysis
# MAGIC 
# MAGIC Before diving into modelling, we want to analyse our collected data - this will inform our feature engineering and modelling processes. Some examples of questions we are looking to address:
# MAGIC 
# MAGIC 1. Are there any missing values: if so we'll need to impute them.
# MAGIC - Are there any highly correlated features? We can consolidate our predictors if so.
# MAGIC - Low/0 variance features: constant values won't be great predictors
# MAGIC - Will we need to scale our values?
# MAGIC - Can we create new features through feature crossing to learn non-linear relationships?

# COMMAND ----------

# DBTITLE 1,Bit of preamble before lift off! 🚀
# MAGIC %run "./Utils/liftoff"

# COMMAND ----------

# check the database in some other ways
print(spark.conf.get("com.databricks.training.spark.dbName"))
print(DATABASE_NAME)

# COMMAND ----------

# DBTITLE 1,Let's peek inside the database we just created specific to your credentials
# recall that `display` gives a nice output format
spark.sql(f"use {DATABASE_NAME}")
display(spark.sql(f"SHOW TABLES"))

# COMMAND ----------

# DBTITLE 1,We now inspect the table of collected experiment data
data = spark.table('phytochemicals_quality')

print(type(data))
# recall that `display` enforces the read
display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC Recall that a predictive model is an approximation to a mathematical function.
# MAGIC 
# MAGIC Can you identify the likely inputs and output of the function?
# MAGIC 
# MAGIC Have you identified a regression or classification problem?

# COMMAND ----------

# DBTITLE 1,As a first step, use Databricks' built in data profiler to examine our dataset.
# use the `+` button to the right of `table` to expose the `Data Profile` tab
# results are approximate
display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC We want to predict the `quality` column, which is a categorial variable with two values. 
# MAGIC 
# MAGIC This makes our problem a binary classifier. Have you heard the phrase "cats n dogs"?

# COMMAND ----------

import pyspark.sql.functions as F
import plotly.express as px
import plotly.io as pio

# COMMAND ----------

# MAGIC %md
# MAGIC Lets first check the spelling of the binary outcomes

# COMMAND ----------

display(data.select('quality').distinct())

# COMMAND ----------

targets = ['Good', 'Bad']
data_targets = {t: data.filter(F.col('quality')==t) for t in targets}

# COMMAND ----------

counts = {t: data_targets[t].count() for t in targets}
print(counts)

print('class imbalance:', counts['Bad']/(counts['Bad'] + counts['Good']))

# COMMAND ----------

# MAGIC %md
# MAGIC The class imbalance shows that the baseline model (predicting every juice to be bad) has 80.3% accuracy

# COMMAND ----------

# A contingency table is an effective visualization for binary classification
data.crosstab("quality","type").show()

# COMMAND ----------

display(data_targets['Good'])

# COMMAND ----------

# DBTITLE 1,Explore Vitamin C vs quality
citric_bad = data.filter(F.col('quality')=='Bad').select(['citric_acid']).toPandas().values.reshape(-1)
citric_good = data.filter(F.col('quality')=='Good').select(['citric_acid']).toPandas().values.reshape(-1)

fig = px.histogram(citric_good)
fig.show()
fig = px.histogram(citric_bad)
fig.show()

# COMMAND ----------

# DBTITLE 1,Your turn: have a play around! 🥳
display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Derived Feature exploration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Runtime derived feature: pH value
# MAGIC From chemistry, we know that pH approximates the concentration of hydrogen ions in a solution. We are going to use this information to include a new (potentially predictive) feature into our model: 
# MAGIC 
# MAGIC $$\\text{pH} = - \\text{log}_{10} ( h_{\\text{concentration}} )$$
# MAGIC $$ \Rightarrow h_{\\text{concentration}} = 10^{-\\text{pH}} $$

# COMMAND ----------

# Using pandas-on-spark, we can use pandas syntax on a distributed spark dataframe
import pyspark.pandas as ps
raw_data = data.to_pandas_on_spark()

# COMMAND ----------

raw_data = raw_data.assign(h_concentration=lambda x: 1/(10**x["pH"]))

# COMMAND ----------

# MAGIC %md
# MAGIC We now look at the distribution of our newly calculated feature - looks good!
# MAGIC 
# MAGIC What we mean by that it "looks good" is that it is approximately normal. 
# MAGIC Later we will scale our numerical variables to have mean zero and unit variance,
# MAGIC which is only meaningful on approximately normal distributions

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context("paper", font_scale=1.8)
sns.displot(raw_data["h_concentration"].to_numpy())
plt.ylabel("Count")
plt.xlabel("hydrogen concentration (moles)")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Runtime derived feature: ratio of acidity to sugar
# MAGIC 
# MAGIC This is a second derived feature which will be computed at runtime.

# COMMAND ----------

raw_data = raw_data.assign(acidity_ratio=lambda x: x["citric_acid"]/x["residual_sugar"])

# COMMAND ----------

sns.displot(raw_data["acidity_ratio"].to_numpy())
plt.ylabel("Count")
plt.xlabel("Acidity ratio (no units)")
plt.show()

# COMMAND ----------

# DBTITLE 1,We apply a log transformation - looks much better!
raw_data = raw_data.assign(acidity_ratio=lambda x: np.log(x["citric_acid"]/x["residual_sugar"]))
sns.displot(raw_data["acidity_ratio"].to_numpy())

plt.ylabel("Count")
plt.xlabel("Acidity ratio (no units)")
plt.show()
