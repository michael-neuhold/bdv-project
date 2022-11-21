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
from pyspark.ml.feature import StringIndexer
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from typing import List

import seaborn as sns

def get_test_value_1():
  return "Hello World - 1"

def get_test_value_2():
  return "Hello World - 2"

def question_alcohol_sex_distribution(data):
  data.filter(data.Dalc > 2).groupBy("sex").count().toPandas().plot.bar(x='sex', y='count', title='Drink a lot of alcohol during the week')
  data.filter(data.Walc > 2).groupBy("sex").count().toPandas().plot.bar(x='sex', y='count', title='Drink a lot of alcohol on the weekend') 

def question_alcohol_week_weekend_distribution(data):
  data.filter(data.Dalc > 2).groupBy("Dalc").count().toPandas().plot.bar(x='Dalc', y='count', title='Drink a lot of alcohol during the week')
  data.filter(data.Walc > 2).groupBy("Walc").count().toPandas().plot.bar(x='Walc', y='count', title='Drink a lot of alcohol on the weekend') 

def question_alcohol_in_freetime(data):
  data.groupBy("freetime").count().toPandas().plot.bar(x='freetime', y='count', title='Alcohol consumption with high and low freetime')

def question_alcohol_health_status(data):
  data.groupBy("health").count().toPandas().plot.bar(x='health', y='count', title='Alcohol consumption correlating with health??')

def question_alcohol_romantic_relationship_status(data):
  data.groupBy("romantic").count().toPandas().plot.bar(x='romantic', y='count', title='Alcohol consumption correlating with romantic relationship status??')

def question_alcohol_parents_education(data):
  data.groupBy("Medu").count().toPandas().plot.bar(x='Medu', y='count', title='Alcohol consuption correlated with mothers education?')
  data.groupBy("Fedu").count().toPandas().plot.bar(x='Fedu', y='count', title='Alcohol consuption correlated with fathers education?') 

def question_health_absences_correlation(data):
  fig = plt.figure(figsize=(20, 8))
  fig.suptitle('Health / Absences')
  for i in range(1, 6):
    fig.add_subplot(1, 5, i)
    plt.title(f'Health {i}')
    sns.boxplot(data=data.filter(data.health == i).select('absences').toPandas())
  
def random_forest_regressor(data: DataFrame, target: str, features: List[str], text_features: List[str] = [],
                            trainings_split: float = 0.8, prepare_features: bool = True, 
                            display_feature_count: int = 10, display_prediction_count: int = 10,
                            max_depth: int = 5, max_bins: int = 32,
                            number_trees: int = 20, feature_subset_strategy: str = 'auto', ):
  # preapre and visualize data
  prepared_data = __prepare_data(data, target, features, text_features, prepare_features, 
                                  display_feature_count, display_prediction_count)

  # split trainings and test data
  train, test = prepared_data.randomSplit([trainings_split, 1 - trainings_split], seed=0)
  regressor = RandomForestRegressor(labelCol=target, featuresCol='prepared_features', 
                                    maxDepth=max_depth, maxBins=max_bins,
                                    numTrees=number_trees, featureSubsetStrategy=feature_subset_strategy)

  # train model
  model = regressor.fit(train)

  # prediction
  prediction = model.transform(test)
  if (display_prediction_count > 0):
    output = prediction \
    .select(['prediction', target, 'prepared_features']) \
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

def random_forest_classifier(data: DataFrame, target: str, features: List[str], text_features: List[str] = [],
                             trainings_split: float = 0.8, prepare_features: bool = True, 
                             display_feature_count: int = 10, display_prediction_count: int = 10,
                             max_depth: int = 5, max_bins: int = 32,
                             number_trees: int = 20, feature_subset_strategy: str = 'auto', ):
  # preapre and visualize data
  prepared_data = __prepare_data(data, target, features, text_features, prepare_features, 
                                  display_feature_count, display_prediction_count)

  # split trainings and test data
  train, test = prepared_data.randomSplit([trainings_split, 1 - trainings_split], seed=0)
  regressor = RandomForestClassifier(labelCol=target, featuresCol='prepared_features', 
                                    maxDepth=max_depth, maxBins=max_bins,
                                    numTrees=number_trees, featureSubsetStrategy=feature_subset_strategy)

  # train model
  model = regressor.fit(train)

  # prediction
  prediction = model.transform(test)
  if (display_prediction_count > 0):
    output = prediction \
    .select(['prediction', target, 'prepared_features']) \
    .limit(display_prediction_count) \
    .toPandas()
    __print_table_('PREDICTIONS', output)

  # evaluation
  y_true = prediction.select([target]).collect()
  y_pred = prediction.select(['prediction']).collect()

  # print evaluation result
  print('EVALUTION RESULTS')
  print(classification_report(y_true, y_pred))

def __prepare_data(data: DataFrame, target: str, features: List[str], text_features: List[str], 
                   prepare_features: bool, display_feature_count: int, display_prediction_count: int):
  # prepare
  prepared_data = data[features + [target]]
  if (prepare_features):
    indexers = list(map(lambda x: StringIndexer(inputCol=x, outputCol='idx_{0}'.format(x)), text_features))
    assembler = VectorAssembler(inputCols=[x for x in features if x not in text_features] + list(map(lambda x: 'idx_{0}'.format(x), text_features)), outputCol='features_assembled')
    scaler = StandardScaler(inputCol='features_assembled', outputCol='prepared_features')
    pipeline = Pipeline(stages=indexers + [assembler, scaler])
    prepared_data = pipeline.fit(prepared_data).transform(prepared_data)

  # show features
  if (display_feature_count > 0):
    output = prepared_data.limit(display_feature_count).toPandas()
    __print_table_('FEATURES', output)

  return prepared_data

def __print_table(title, data):
    print(title)
    print(data.to_markdown())
    print()