# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC # Let the Machines Learn!

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="float:right">
# MAGIC   <img src="https://ajmal-field-demo.s3.ap-southeast-2.amazonaws.com/apj-sa-bootcamp/machine_learning_model.png" width="1000px">
# MAGIC </div>
# MAGIC 
# MAGIC 
# MAGIC We are going to train an a model to predict the quality of an orange given the chemical makeup of the orange. This will help us find the key indicators of quality.
# MAGIC The key indicators can then be used engineer a great orange! We will begin as follows:
# MAGIC 
# MAGIC 1. The feature set will then be uploaded to the Feature Store.
# MAGIC 2. We will pre-process our numerical and categorial column(s).
# MAGIC 3. We will train multiple models from our ML runtime and assess the best model.
# MAGIC 4. Machine learning model metrics, parameters, and artefacts will be tracked with mlflow.
# MAGIC 5. 🚀 Deployment of our model!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Data

# COMMAND ----------

import pyspark.sql.functions as F
import numpy as np

# COMMAND ----------

# MAGIC %run "./Utils/Fetch_User_Metadata"

# COMMAND ----------

# DBTITLE 1,Note: our original data set does not have our pre-computed derived features
spark.sql(f"USE {DATABASE_NAME}")
data = spark.table("phytochemicals_quality")
display(data)

# COMMAND ----------

# DBTITLE 1,We need a FeatureLookup to pull our pre-computed features from our feature store
from databricks.feature_store import FeatureLookup, FeatureStoreClient

fs = FeatureStoreClient()

feature_table = f"{DATABASE_NAME}.features_oj_prediction_experiment"

feature_lookup = FeatureLookup(
  table_name=feature_table,
  feature_names=['total_sulfur_dioxide_avg', "fixed_acidity_avg", "total_sulfur_dioxide_std", "fixed_acidity_std"],
  lookup_key = ["type"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/valid/test split

# COMMAND ----------

# MAGIC %md
# MAGIC Recall that our pre-computed features in the feature store, were computed from only the training data.

# COMMAND ----------

# DBTITLE 1,Read in our split data
train_data = spark.table('train_data');
valid_data = spark.table('valid_data');
test_data = spark.table('test_data');

# COMMAND ----------

# DBTITLE 1,We generate a training and validation data set using our feature lookups
training_data = fs.create_training_set(
  df=train_data,
  feature_lookups=[feature_lookup],
  label = 'quality',
  exclude_columns="customer_id"
).load_df()

validation_data = fs.create_training_set(
  df=valid_data,
  feature_lookups=[feature_lookup],
  label = 'quality',
  exclude_columns="customer_id"
).load_df()

# COMMAND ----------

# check that customer_id has been dropped
"customer_id" in training_data.columns

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC # Machine Learning Pipeline
# MAGIC We will apply different transformations for numerical and categorical columns. 
# MAGIC 
# MAGIC 
# MAGIC <div style="float:right">
# MAGIC   <img src="https://www.pngkit.com/png/full/37-376558_pipe-8-bit-mario.png" width="800px">
# MAGIC </div>
# MAGIC 
# MAGIC 
# MAGIC For numerical columns, we will:
# MAGIC - Impute missing values with the mean of the column.
# MAGIC - Scale numerical values to 0 mean and unit variance to reduce the impact of larger values.
# MAGIC 
# MAGIC For categorical columns:
# MAGIC - Impute missing values with empty strings
# MAGIC - One hot encode the type of orange

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initial Preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC This includes processing which will not need to be performed on new data but it needed for the training phase
# MAGIC 
# MAGIC All processing which must be performed on new data, should be in the pipeline.

# COMMAND ----------

# DBTITLE 1,Reformat target variable as binary integer
training_data = training_data.withColumn('quality', (F.col('quality') == "Good").cast('int'))
validation_data = validation_data.withColumn('quality', (F.col('quality') == "Good").cast('int'))

# COMMAND ----------

# DBTITLE 1,Make list of numerical and categorical features
# convert top of the spark df to pandas, the use native pandas fucntion to get numerical columns
# we have to remove the label column
numerical_features = training_data.limit(1).toPandas()._get_numeric_data().columns.tolist()
numerical_features = [c for c in numerical_features if c != 'quality']

# if you're not numerical, you're categorical
categorical_features = list(set(training_data.columns) - set(numerical_features))
categorical_features = [c for c in categorical_features if c != 'quality']

# The only categorical feature is the type of the orange-juice (we removed customer_id when we set up the feature store lookup)
print('categorical_features:', categorical_features)

# COMMAND ----------

'quality' in numerical_features

# COMMAND ----------

# MAGIC %md
# MAGIC ### 💾 Persist training/valid data
# MAGIC Save into our database for future use Just to be clear, these tables have the precomputed derived features and have been preprocessed. We have not yet done things like imputation, scaling and runtime features. These steps will be in the pipeline which we study next. 

# COMMAND ----------

def save_to_db(df, name, pandas=False):
  if pandas:
    df = spark.createDataFrame(df)
  
  (df
        .write
        .mode("overwrite")
        .format("delta")
        .saveAsTable(f"{DATABASE_NAME}.{name}")
  )

save_to_db(training_data, "training_data")
save_to_db(validation_data, "validation_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline
# MAGIC 
# MAGIC **Linear pipeline**: takes a dataframe, applies a linear series of transformations then an estimation/prediction
# MAGIC 
# MAGIC **Non-Linear Pipieline**: takes a dataframe, applies transformations in a Directed Acyclic Graph then an estimation/prediction
# MAGIC 
# MAGIC The pipeline steps are carried out in the same order of the index of the list.

# COMMAND ----------

# MAGIC %md
# MAGIC ### String Indexer

# COMMAND ----------

# MAGIC %md
# MAGIC The purpose of a string indexer is to convert a categorical variable to a  
# MAGIC numerical variable.
# MAGIC 
# MAGIC For example:
# MAGIC  
# MAGIC | col |  
# MAGIC |-----|  
# MAGIC |  a  |
# MAGIC |  b  |
# MAGIC |  c  |
# MAGIC 
# MAGIC becomes
# MAGIC  
# MAGIC | col |  
# MAGIC |-----|  
# MAGIC |  1  |
# MAGIC |  2  |
# MAGIC |  3  |

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

df = spark.createDataFrame([('a',), ('b',), ('c',), ('d',)], ["col"])
df = StringIndexer(inputCol="col", outputCol="col_num").fit(df).transform(df)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Do you see anything conceptually wrong with using a string-indexed column for predicitive modelling?

# COMMAND ----------

# MAGIC %md
# MAGIC ### One Hot Encoders

# COMMAND ----------

# MAGIC %md
# MAGIC To avoid sequential-categories, we have to one-hot encode.
# MAGIC 
# MAGIC One-hot encoding means that each column will be transformed into multiple columns.  
# MAGIC The number of columns is equal to the number of distinct entries in the original column.  
# MAGIC This can lead to a huge number of columns and potentially a degredaton of model peformance.
# MAGIC 
# MAGIC For example:
# MAGIC  
# MAGIC | col |  
# MAGIC |-----|  
# MAGIC |  1  |
# MAGIC |  2  |
# MAGIC |  3  |
# MAGIC 
# MAGIC 
# MAGIC becomes
# MAGIC | col1 | col2 | col3 |
# MAGIC |------|------|------|  
# MAGIC |  1   |  0   |   0  |
# MAGIC |  0   |  1   |   0  |
# MAGIC |  0   |  0   |   1  |

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder

df = spark.createDataFrame([(1.0,), (2.0,), (3.0,), (4.0,), (5.0,)], ["c"])
df = OneHotEncoder(inputCol="c", outputCol="c_onehot").fit(df).transform(df)
df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC To understand the `c_onehot` column, we need to understand the sparse-vector type in spark
# MAGIC 
# MAGIC (i, [a1, b1, c1,...], [a2, b2, c2,...])
# MAGIC 
# MAGIC m - length of the vector  
# MAGIC a1, a2,..., am - positions where a non-zero entry exists  
# MAGIC b1, b2,..., bm - values at the above positions

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imputing
# MAGIC There will always be missing values, we need to impute them.

# COMMAND ----------

# DBTITLE 1,Categorical Imputer
# I am not aware of a native categorical imputer in spark.
# This is one way to build a custom transformer
#
# By extending DefaultParamsWritable, DefaultParamsReadable, we allow the pipeline to be serialized.
# This is an example of something in the Spark stack whcih was initially only available in Scala 
# and was later ported to PySpark.

from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable

class CategoricalImputer(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    def __init__(self, categorical_columns, fillValue=""):
        super(CategoricalImputer, self).__init__()
        self.categorical_columns = categorical_columns
        self.fillValue = fillValue
        
    def _transform(self, df):
        df = df.na.fill(self.fillValue, subset=self.categorical_columns)
        return df
      
df = spark.createDataFrame([('a',), ('b',), (None,)], ["col"])
display(CategoricalImputer(categorical_columns=['col'], fillValue='xxx').transform(df))

# COMMAND ----------

# DBTITLE 1,Numerical Imputer
from pyspark.ml.feature import Imputer

df = spark.createDataFrame([(1,),(3,),(3,),(None,)], ['col'])

Imputer(inputCols=['col'], outputCols=['col'], strategy='median').fit(df).transform(df).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scaling

# COMMAND ----------

# MAGIC %md
# MAGIC It is useful to rescale numerical columns to have a zero mean and unit variance.
# MAGIC 
# MAGIC A curious thing to take note of is that in pyspark, StandardScaler requires vector inputs. StandardScaler takes just a single column and this column must be of type `vector`. There is a transformer, the `VectorAssembler` which can vectorize multiple columns.

# COMMAND ----------

# DBTITLE 0,Scaling for Multiple Columns
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml import Pipeline

df = spark.createDataFrame([(1,1),(3,3),(3,4)], ['col1', 'col2'])

assembler = VectorAssembler(inputCols=df.columns, outputCol="x_vec")
scalar = StandardScaler(withMean=True, inputCol="x_vec", outputCol="x_sc")

pipeline = Pipeline(stages=[assembler, scalar])
display(pipeline.fit(df).transform(df))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Runtime Derived Features

# COMMAND ----------

class AcidityRatioTransformer(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    def __init__(self, ratio_colname, acid_colname, sugar_colname):
        super(AcidityRatioTransformer, self).__init__()
        self.acid_colname = acid_colname
        self.sugar_colname = sugar_colname
        self.ratio_colname = ratio_colname

    def _transform(self, df):
        df = df.withColumn(self.ratio_colname, F.log(F.col(self.acid_colname)/F.col(self.sugar_colname)) )
        return df
      
      
class HConTransformer(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    def __init__(self, hconc_colname, ph_colname):
        super(HConTransformer, self).__init__()
        self.hconc_colname = hconc_colname
        self.ph_colname = ph_colname

    def _transform(self, df):
        df = df.withColumn(self.hconc_colname, F.pow(10, - F.col(self.ph_colname)))
        return df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical Pipeline
# MAGIC 
# MAGIC We have three transformers in our categorical pipeline
# MAGIC 1. imputer
# MAGIC 2. string indexer
# MAGIC 3. one hot encoder

# COMMAND ----------

# note that the suffix _oh is short for one-hot

from pyspark.ml import Pipeline

onehot_features = [c + "_oh" for c in categorical_features]

categorical_imputer = CategoricalImputer(categorical_columns=categorical_features, fillValue="")

string_indexer = StringIndexer(inputCols=categorical_features, outputCols=[c + "_num" for c in categorical_features])

onehot_encoder = OneHotEncoder(inputCols=[c + "_num" for c in categorical_features], outputCols=onehot_features)

# Put together in a pipeline
categorical_pipeline = Pipeline(stages=[categorical_imputer, string_indexer, onehot_encoder])

# COMMAND ----------

# as a demonstration we fit and transform the pipeline but in the real thing, we must be a bit more careful
display(categorical_pipeline.fit(validation_data).transform(validation_data))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical Pipeline
# MAGIC 
# MAGIC There are three transformations in our numerical pipeline
# MAGIC 1. runtime derived features
# MAGIC 2. imputation
# MAGIC 3. vectorization
# MAGIC 4. scaling

# COMMAND ----------

from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler

# 1. The runtime derived features
acidity_transformer = AcidityRatioTransformer("acidity_ratio", 'citric_acid', 'residual_sugar')
hcon_transformer = HConTransformer('h_concentration', "pH")

# We mustn't forget to include the derived numerical features in the list of features to be scaled
derived_numerical_features = numerical_features + ['acidity_ratio', 'h_concentration']

# 2. imputation
numerical_imputer = Imputer(inputCols=derived_numerical_features, outputCols=derived_numerical_features, strategy='median')

# 3. Vectorization
numerical_vectorizer = VectorAssembler(inputCols=derived_numerical_features, outputCol='numerical_vec')

# 4. Scaling
numerical_scalar = StandardScaler(withMean=True, inputCol="numerical_vec", outputCol="numerical_scaled")

# We must impute after computing the derived features
numerical_pipeline = Pipeline(stages=[acidity_transformer, hcon_transformer, numerical_imputer, numerical_vectorizer, numerical_scalar])

# COMMAND ----------

numerical_pipeline.fit(validation_data).transform(validation_data).toPandas()[:2]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Processing Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC We combine the categorical and numerical pipelines

# COMMAND ----------

# This final assmebler combines the categorical and numerical columns into one
final_assembler = VectorAssembler(inputCols=onehot_features+['numerical_scaled'], outputCol="features")

pipeline = Pipeline(stages = [categorical_pipeline, numerical_pipeline, final_assembler])

# COMMAND ----------

pipeline.fit(training_data).transform(validation_data).toPandas()[:2]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Estimator

# COMMAND ----------

# MAGIC %run "./Utils/ml_utils"

# COMMAND ----------

import mlflow

PROJECT_PATH = "/Users/nick.halmagyi@versor.com.au/MLFlowExperiments"
experiment_name = "Orange-Quality-Prediction"
experiment_path = os.path.join(PROJECT_PATH, experiment_name)
experiment_id = '585900274276948'

# experiment_id = mlflow.create_experiment(experiment_path)
mlflow.set_experiment(experiment_path)

mlflow.spark.autolog()

# COMMAND ----------

# DBTITLE 1,Our pipeline needs to end in an estimator, we will use a random forest
from pyspark.ml.classification import RandomForestClassifier

rf_params = {'numTrees':500, 
             'maxDepth':10, 
             'featuresCol':'features',
             'labelCol': "quality", 
             'seed':42}

rf = RandomForestClassifier(**rf_params)

model = Pipeline(stages=[pipeline, rf])

# COMMAND ----------

import os

model = Pipeline(stages=[pipeline, rf])

model_dir = f'/dbfs/FileStore/{USERNAME}/models'
dbutils.fs.mkdirs(model_dir)

model_name = 'juice_random_forest.model'
path = os.path.join(model_dir, model_name)

dbutils.fs.rm(path, recurse=True)
model.save(path)

# COMMAND ----------

baseline_predictions = training_data.select('quality').withColumn('prediction', F.lit(1).cast('double'))  
baseline_evaluations = make_binary_evaluation(baseline_predictions, labelCol='quality', predCol='prediction')

# COMMAND ----------

with mlflow.start_run(run_name="random_forest_pipeline",
                      experiment_id=experiment_id) as mlflow_run:
    
  
  model = Pipeline(stages=[pipeline, rf])
  
  model = model.fit(training_data)
  
  predictions = model.transform(validation_data).select('quality', 'prediction')
  
  metrics = make_binary_evaluation(predictions, labelCol='quality', predCol='prediction')
  
  mlflow.log_params(rf_params)
  mlflow.log_metrics(metrics)

# COMMAND ----------

import os

model_dir = f'/dbfs/FileStore/{USERNAME}/models'
dbutils.fs.mkdirs(model_dir)

model_name = 'juice_random_forest.model'
path = os.path.join(model_dir, model_name)

dbutils.fs.rm(path, recurse=True)
model.save(path)

# COMMAND ----------

baseline_predictions = predictions.select('quality').withColumn('prediction', F.lit(1).cast('double'))
make_binary_evaluation(baseline_predictions, labelCol='quality', predCol='prediction')

# COMMAND ----------

make_binary_evaluation(predictions, labelCol='quality', predCol='prediction')

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

# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="float:right">
# MAGIC   <img src="https://ajmal-field-demo.s3.ap-southeast-2.amazonaws.com/apj-sa-bootcamp/shap_logged.gif" width="600px">
# MAGIC </div>
# MAGIC 
# MAGIC 
# MAGIC ## Logging other artefacts in runs
# MAGIC 
# MAGIC We have flexibility over the artefacts we want to log. By logging artefacts with runs we have examine the quality of fit to better determine if we have overfit or if we need to retrain, etc. These artefacts also help with reproducibility.
# MAGIC 
# MAGIC As an example, let's log the partial dependence plot from SHAP with a single model run. 👇

# COMMAND ----------

def generate_shap_plot(model, data):
  import shap
  global image
  sample_data = data.sample(n=100)
  explainer = shap.TreeExplainer(model["classifier"])
  shap_values = explainer.shap_values(model["preprocessor"].transform(sample_data))
  
  fig = plt.figure(1)
  ax = plt.gca()
  
  shap.dependence_plot("rank(1)", shap_values[0],
                       model["preprocessor"].transform(sample_data),
                       ax=ax, show=False)
  plt.title(f"Acidity dependence plot")
  plt.ylabel(f"SHAP value for the Acidity")
  image = fig
  # Save figure
  fig.savefig(f"/dbfs/FileStore/{USERNAME}_shap_plot.png")

  # Close plot
  plt.close(fig)
  return image

# COMMAND ----------

# DBTITLE 1,We now log our SHAP image as an artefact within this run
# Enable automatic logging of input samples, metrics, parameters, and models
import matplotlib.pyplot as plt

with mlflow.start_run(run_name="random_forest_pipeline_2", experiment_id=experiment_id) as mlflow_run:
    # Fit our estimator
    rf_model.fit(X_training, y_training)
    
    # Log our parameters
    mlflow.log_params(rf_params)
    
    # Training metrics are logged by MLflow autologging
    # Log metrics for the validation set
    mlflow.sklearn.eval_and_log_metrics(rf_model,
                                        X_validation,
                                        y_validation,
                                        prefix="val_")
    shap_fig = generate_shap_plot(rf_model, X_validation)
    mlflow.log_artifact(f"/dbfs/FileStore/{USERNAME}_shap_plot.png")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## Search best hyper parameters with HyperOpt (Bayesian optimization) accross multiple nodes
# MAGIC <div style="float:right"><img src="https://quentin-demo-resources.s3.eu-west-3.amazonaws.com/images/bayesian-model.png" style="height: 330px"/></div>
# MAGIC Our model performs well but we want to run hyperparameter optimisation across our parameter search space. We will use HyperOpt to do so. For fun, we're going to try an XGBoost model here.
# MAGIC 
# MAGIC This model is a good start, but now we want to try multiple hyper-parameters and search the space rather than a fixed size.
# MAGIC 
# MAGIC GridSearch could be a good way to do it, but not very efficient when the parameter dimension increase and the model is getting slow to train due to a massive amount of data.
# MAGIC 
# MAGIC HyperOpt search accross your parameter space for the minimum loss of your model, using Baysian optimization instead of a random walk

# COMMAND ----------

# MAGIC %md
# MAGIC ![my_test_image](https://www.jeremyjordan.me/content/images/2017/11/grid_search.gif)
# MAGIC ![my_test_image](https://www.jeremyjordan.me/content/images/2017/11/Bayesian_optimization.gif)

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope


search_space = {
  'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
  'colsample_bytree': hp.uniform('colsample_bytree', 0.01, 1), 
  'learning_rate': hp.loguniform('learning_rate', 0.01, 1),
  'n_estimators': scope.int(hp.quniform('n_estimators', 100, 1000, 10)),
  'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
  'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
  'min_child_weight': hp.loguniform('min_child_weight', -1, 6),
  'objective': 'binary:logistic',
  'verbosity': 0,
  'n_jobs': -1
}

# COMMAND ----------

# DBTITLE 1,We begin by defining our objective function
# With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
from mlflow.models.signature import infer_signature
from xgboost import XGBClassifier
import mlflow.xgboost
from sklearn.metrics import roc_auc_score


def f_train(trial_params):
  with mlflow.start_run(nested=True):
    mlflow.xgboost.autolog(log_input_examples=True, silent=True)
    
    # Redefine our pipeline with the new params from the search algo    
    classifier = XGBClassifier(**trial_params)
    
    xgb_model = Pipeline([
      ("preprocessor", preprocessor),
      ("classifier", classifier)
    ])
    
    # Fit, predict, score
    xgb_model.fit(X_training, y_training)
    predictions_valid = xgb_model.predict(X_validation)
    auc_score = roc_auc_score(y_validation.values, predictions_valid)
    
    # Log :) 
    signature = infer_signature(X_training, xgb_model.predict(X_training))
    mlflow.log_metric('auc', auc_score)

    # Set the loss to -1*auc_score so fmin maximizes the auc_score
    return {'status': STATUS_OK, 'loss': -1*auc_score}

# COMMAND ----------

# Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep. 
# A reasonable value for parallelism is the square root of max_evals.
spark_trials = SparkTrials(parallelism=10)

# Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
with mlflow.start_run(run_name='xgboost_models', experiment_id=experiment_id) as mlflow_run:
  best_params = fmin(
    fn=f_train, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=5,
    trials=spark_trials, 
  )
