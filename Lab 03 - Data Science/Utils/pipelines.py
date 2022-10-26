# Databricks notebook source
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsWritable, DefaultParamsReadable

class CategoricalImputer(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    def __init__(self, categorical_columns=[], fillValue=""):
        super(CategoricalImputer, self).__init__()
        self.categorical_columns = categorical_columns
        self.fillValue = fillValue
        
    def _transform(self, df):
        df = df.na.fill(self.fillValue, subset=self.categorical_columns)
        return df
      
      
class AcidityRatioTransformer(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    def __init__(self, ratio_colname='acidity_ratio', acid_colname='citric_acid', sugar_colname='residual_sugar'):
        super(AcidityRatioTransformer, self).__init__()
        self.acid_colname = acid_colname
        self.sugar_colname = sugar_colname
        self.ratio_colname = ratio_colname

    def _transform(self, df):
        df = df.withColumn(self.ratio_colname, F.log(F.col(self.acid_colname)/F.col(self.sugar_colname)) )
        return df
      
      
class HConTransformer(Transformer, DefaultParamsWritable, DefaultParamsReadable):
    def __init__(self, hconc_colname='h_concentration', ph_colname='pH'):
        super(HConTransformer, self).__init__()
        self.hconc_colname = hconc_colname
        self.ph_colname = ph_colname

    def _transform(self, df):
        df = df.withColumn(self.hconc_colname, F.pow(10, - F.col(self.ph_colname)))
        return df
