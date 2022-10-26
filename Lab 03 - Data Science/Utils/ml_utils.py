# Databricks notebook source
import pyspark.sql.functions as F

def accuracy(DF, labelCol='label', predCol='prediction'):
  DF = DF.withColumn('acc', (F.col(labelCol)==F.col(predCol)).cast('int'))
  acc = DF.agg({'acc': 'mean'}).collect()[0]['avg(acc)']
  return acc

def true_positives(DF, labelCol='label', predCol='prediction'):
  DF = DF.where(F.col(labelCol)==1)
  DF = DF.withColumn('tp', (F.col(labelCol)==F.col(predCol)).cast('int'))
  tp = DF.agg({'tp': 'mean'}).collect()[0]['avg(tp)']
  return tp

def true_negatives(DF, labelCol='label', predCol='prediction'):
  DF = DF.where(F.col(labelCol)==0)
  DF = DF.withColumn('tn', (F.col(labelCol)==F.col(predCol)).cast('int'))
  tn = DF.agg({'tn': 'mean'}).collect()[0]['avg(tn)']
  return tn

def false_positives(DF, labelCol='label', predCol='prediction'):
  DF = DF.where(F.col(labelCol)==1)
  DF = DF.withColumn('fp', (F.col(labelCol)!=F.col(predCol)).cast('int'))
  fp = DF.agg({'fp': 'mean'}).collect()[0]['avg(fp)']
  return fp

def false_negatives(DF, labelCol='label', predCol='prediction'):
  DF = DF.where(F.col(labelCol)==0)
  DF = DF.withColumn('fn', (F.col(labelCol)!=F.col(predCol)).cast('int'))
  fn = DF.agg({'fn': 'mean'}).collect()[0]['avg(fn)']
  return fn


def make_binary_evaluation(predictions, labelCol='label', predCol='prediction'):
  acc = accuracy(predictions, labelCol=labelCol, predCol='prediction')
  tp = true_positives(predictions, labelCol=labelCol, predCol=predCol)
  tn = true_negatives(predictions, labelCol=labelCol, predCol=predCol)
  fp = false_positives(predictions, labelCol=labelCol, predCol=predCol)
  fn = false_negatives(predictions, labelCol=labelCol, predCol=predCol)
  
  from pyspark.ml.evaluation import BinaryClassificationEvaluator
  evaluator = BinaryClassificationEvaluator()
  evaluator.setRawPredictionCol(predCol).setLabelCol(labelCol)

  areaUnderPR = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
  areaUnderROC = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
  
  return {'acc': acc, 
          'tp':tp, 'tn': tn, 'fp': fp, 'fn': fn, 
         'areaUnderPR': areaUnderPR,
         'areaUnderROC': areaUnderROC}
