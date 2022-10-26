# Databricks notebook source
# MAGIC %md
# MAGIC #Training the ML Model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Data

# COMMAND ----------

# MAGIC %run "./Utils/Fetch_User_Metadata"

# COMMAND ----------

# MAGIC %run "./Utils/pipelines"

# COMMAND ----------

# MAGIC %run "./Utils/ml_utils"

# COMMAND ----------

spark.sql(f"USE {DATABASE_NAME}");

# COMMAND ----------

import os
from pyspark.ml import Pipeline

model_dir = f'/dbfs/FileStore/{USERNAME}/models'
model_name = 'juice_processing.pipeline'
path = os.path.join(model_dir, model_name)

processing_pipeline = Pipeline.load(path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Estimator

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

rf_params = {'numTrees':500, 
             'maxDepth':10, 
             'featuresCol':'features',
             'labelCol': "quality", 
             'seed':42}

rf = RandomForestClassifier(**rf_params)

final_pipeline = Pipeline(stages=[processing_pipeline, rf])

# COMMAND ----------

# MAGIC %md
# MAGIC Note that final_pipeline is of type  
# MAGIC `<class 'pyspark.ml.pipeline.Pipeline'>`  
# MAGIC When the pipeline has been fit, it will be of type  
# MAGIC `<class 'pyspark.ml.pipeline.PipelineModel'>`  

# COMMAND ----------

# Grad the unprocessed data
training_data = spark.table('training_data')
validation_data = spark.table('validation_data')

# COMMAND ----------

# MAGIC %md
# MAGIC It is important to understand a baseline prediction

# COMMAND ----------

baseline_predictions = training_data.select('quality').withColumn('prediction', F.lit(1).cast('double'))  
baseline_evaluations = make_binary_evaluation(baseline_predictions, labelCol='quality', predCol='prediction')
baseline_evaluations

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## Create an Experiment Manually 👩‍🔬
# MAGIC 
# MAGIC <div style="float:right">
# MAGIC   <img src="https://ajmal-field-demo.s3.ap-southeast-2.amazonaws.com/apj-sa-bootcamp/create_experiment.gif" width="800px">
# MAGIC </div>
# MAGIC 
# MAGIC 
# MAGIC We are now going to manually create an experiment using our UI. To do this, we will follow the following steps:
# MAGIC 
# MAGIC 0. Ensure that you are in the Machine Learning persona by checking the LHS pane and ensuring it says **Machine Learning**.
# MAGIC - Click on the ```Experiments``` button.
# MAGIC - Click the "Create an AutoML Experiment" arrow dropdown
# MAGIC - Press on **Create Blank Experiment**
# MAGIC - Put the experiment name as: "**first_name last_name Orange Quality Prediction**", so e.g. "Ajmal Aziz Orange Quality Prediction"

# COMMAND ----------

import mlflow

PROJECT_PATH = "/Users/nick.halmagyi@versor.com.au/MLFlowExperiments"
experiment_name = "Orange-Quality-Prediction"
experiment_path = os.path.join(PROJECT_PATH, experiment_name)
experiment_id = '585900274276948'

mlflow.set_experiment(experiment_path)

mlflow.spark.autolog()

# COMMAND ----------

with mlflow.start_run(run_name="random_forest_pipeline",
                      experiment_id=experiment_id) as mlflow_run:
    
    
  model = final_pipeline.fit(training_data)
  
  predictions = model.transform(validation_data).select('quality', 'prediction')
  
  metrics = make_binary_evaluation(predictions, labelCol='quality', predCol='prediction')
  
  mlflow.log_params(rf_params)
  mlflow.log_metrics(metrics)

# COMMAND ----------

metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyper-parameter tuning

# COMMAND ----------

# MAGIC %md
# MAGIC Some good references from Databricks are [here](https://docs.databricks.com/_static/notebooks/hyperopt-spark-ml.html) and [here](https://www.databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html)

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope


search_space = {
  'maxDepth': scope.int(hp.quniform('max_depth', 10, 20, 1)),
  'numTrees': scope.int(hp.quniform('n_estimators', 800, 900, 10))
}

spark_trials = Trials()
# spark_trials = SparkTrials(parallelism=10)

# COMMAND ----------

def f_train(trial_params):
  with mlflow.start_run(nested=True):

    # with this ordering, the values in trial_params will overwrite those in rf_params
    trial_params = {**rf_params, **trial_params} 
    
    rf = RandomForestClassifier(**trial_params)
    final_pipeline = Pipeline(stages=[processing_pipeline, rf])
    model = final_pipeline.fit(training_data)

    predictions = model.transform(validation_data).select('quality', 'prediction')

    metrics = make_binary_evaluation(predictions, labelCol='quality', predCol='prediction')

    mlflow.log_params(rf_params)
    mlflow.log_metrics(metrics)
    
    # Set the loss to -1*auc_score so fmin maximizes the auc_score
    return {'status': STATUS_OK, 'loss': -1 * metrics['areaUnderROC']}

# COMMAND ----------

with mlflow.start_run(run_name='rf_models', experiment_id=experiment_id) as mlflow_run:
  
  best_params = fmin(
    fn=f_train, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=5
  )

# COMMAND ----------


