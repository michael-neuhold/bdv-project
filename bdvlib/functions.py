# import the necessary packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import cv2

from math import sqrt, ceil, log10, floor
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import dendrogram

import sys
import os
from os import path
import struct
from array import array

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from typing import List

def get_test_value_1():
  return "Hello World - 1"

def get_test_value_2():
  return "Hello World - 2"

def question_alcohol_sex_distribution(data):
  data.filter(data.Dalc > 2).groupBy("sex").count().toPandas().plot.bar(x='sex', y='count', title='Drink a lot of alcohol during the week')
  data.filter(data.Walc > 2).groupBy("sex").count().toPandas().plot.bar(x='sex', y='count', title='Drink a lot of alcohol on the weekend') 

def random_forest_regressor(data: DataFrame, target: str, features: List[str],
                            trainings_split: float = 0.8, scale_features: bool = True, 
                            display_feature_count: int = 10, display_prediction_count: int = 10,
                            max_depth: int = 5, max_bins: int = 32,
                            number_trees: int = 20, feature_subset_strategy: str = 'auto', ):
  # prepare data
  extracted_data = data[features + [target]]
  assembler = VectorAssembler(inputCols=features, outputCol='features')
  prepared_data = assembler.transform(extracted_data)

  # scale features
  if (scale_features):
    scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures')
    prepared_data = scaler.fit(prepared_data).transform(prepared_data)

  # show features
  if (display_feature_count > 0):
    output = prepared_data.limit(display_feature_count).toPandas()
    __print_table('FEATURES', output)

  # split trainings and test data
  train, test = prepared_data.randomSplit([trainings_split, 1 - trainings_split], seed=0)
  regressor = RandomForestRegressor(labelCol=target, featuresCol='scaledFeatures', 
                                    maxDepth=max_depth, maxBins=max_bins,
                                    numTrees=number_trees, featureSubsetStrategy=feature_subset_strategy)

  # train model
  model = regressor.fit(train)

  # prediction
  prediction = model.transform(test)
  if (display_prediction_count > 0):
    output = prediction \
    .select(['prediction', target, 'scaledFeatures']) \
    .limit(display_prediction_count) \
    .toPandas()
    __print_table('PREDICTIONS', output)

  # evaluation
  evaluator = RegressionEvaluator(predictionCol='prediction', labelCol=target)  
  rmse = evaluator.setMetricName("rmse").evaluate(prediction)
  mse = evaluator.setMetricName("rmse").evaluate(prediction)
  mae = evaluator.setMetricName("mae").evaluate(prediction)
  r2 = evaluator.setMetricName("r2").evaluate(prediction)

  # print evaluation result
  print('EVALUTION RESULTS')
  print(f'Root Mean Squared Error (RMSE) on test data = {rmse}')
  print(f'Mean squared error (MSE) on test data = {mse}')
  print(f'Regression through the origin(R2) on test data = {r2}')
  print(f'Mean absolute error (MAE) on test data = {mae}')

def __print_table(title, data):
    print(title)
    print(data.to_markdown())
    print()

def __print_table(title, data):
    print(title)
    print(data.to_markdown())
    print()