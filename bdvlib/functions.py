# import the necessary packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, r2_score, classification_report
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
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import DataFrame
from typing import List

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def correlate_class_boxplot(data: DataFrame, xAttr: str, yAttr: str, first: int, last: int, title: str = None, size = (15, 5)):
  fig = plt.figure(figsize=size)
  if title == None:
    title = f'{xAttr.capitalize()} / {yAttr.capitalize()}'
  fig.suptitle(title)
  for i in range(first, last + 1):
    fig.add_subplot(1, last - first + 1, i if first > 0 else i + 1)
    plt.title(f'{xAttr} {i}')
    if ylim != None:
      plt.ylim(ylim[0], 40)
    sns.boxplot(data=data.filter(getattr(data, xAttr) == i).select(yAttr).toPandas())

def question_alcohol_sex_distribution(data):
  plt.figure()
  plt.title('Alcohol consumption correlating with sex')
  sns.barplot(data=data.select('Dalc', 'sex').groupBy('Dalc', 'sex').count().toPandas(), x="Dalc", y='count', hue="sex", palette=['red', 'blue'], alpha=0.75)
  ax = sns.barplot(data=data.select('Walc', 'sex').groupBy('Walc', 'sex').count().toPandas(), x="Walc", y='count', hue="sex", palette=['orange', 'violet'], alpha=0.75)
  h, l = ax.get_legend_handles_labels()
  ax.legend(h, ['Dalc M', 'Dalc F', 'Walc M', 'Walc F'])
  plt.xlabel('Dalc/Walc')
  plt.show()

def question_alcohol_week_weekend_distribution(data):
  plt.figure()
  plt.title('Alcohol consumption correlating with sex')
  sns.barplot(data=data.select('Dalc').groupBy('Dalc').count().toPandas(), x="Dalc", y='count', color='red', alpha=0.75)
  sns.barplot(data=data.select('Walc').groupBy('Walc').count().toPandas(), x="Walc", y='count', color='green', alpha=0.75)
  red_patch = mpatches.Patch(color='red', label='Dalc')
  green_patch = mpatches.Patch(color='green', label='Walc')
  plt.legend(handles=[red_patch, green_patch])
  plt.xlabel('Dalc/Walc')
  plt.show()

def question_alcohol_in_freetime(data):
  plt.figure()
  plt.title('Alcohol consumption correlating with freetime')
  sns.barplot(data=data.select('Dalc', 'freetime').groupBy('Dalc', 'freetime').count().toPandas(), x="freetime", y='count', hue="Dalc")
  plt.show()
  plt.figure()
  plt.title('Alcohol consumption correlating with freetime')
  sns.barplot(data=data.select('Walc', 'freetime').groupBy('Walc', 'freetime').count().toPandas(), x="freetime", y='count', hue="Walc")
  plt.show()

def question_alcohol_health_status(data):
  plt.figure()
  sns.barplot(data=data.select('Dalc', 'health').groupBy('Dalc', 'health').count().toPandas(), x="health", y='count', hue="Dalc")
  plt.show()
  plt.figure()
  sns.barplot(data=data.select('Dalc', 'health').groupBy('Dalc', 'health').count().toPandas(), x="Dalc", y='count', hue="health")
  plt.show()

def question_alcohol_romantic_relationship_status(data):
  plt.figure()
  plt.title('Alcohol consumption correlating with romantic relationship status')
  sns.barplot(data=data.select('Dalc', 'romantic').groupBy('Dalc', 'romantic').count().toPandas(), x="Dalc", y='count', hue="romantic", palette=['red', 'blue'], alpha=0.75)
  ax = sns.barplot(data=data.select('Walc', 'romantic').groupBy('Walc', 'romantic').count().toPandas(), x="Walc", y='count', hue="romantic", palette=['orange', 'violet'], alpha=0.75)
  h, l = ax.get_legend_handles_labels()
  ax.legend(h, ['Dalc no', 'Dalc yes', 'Walc no', 'Walc yes'])
  plt.xlabel('Dalc/Walc')
  plt.show()

def question_alcohol_parents_education(data):
  plt.figure()
  sns.barplot(data=data.select('Dalc', 'Medu').groupBy('Medu').mean('Dalc').toPandas(), x='Medu', y='avg(Dalc)')
  plt.show()
  plt.figure()
  sns.barplot(data=data.select('Dalc', 'Medu').groupBy('Dalc', 'Medu').count().toPandas(), x="Medu", y='count', hue="Dalc")
  plt.show()
  plt.figure()
  sns.barplot(data=data.select('Dalc', 'Fedu').groupBy('Fedu').mean('Dalc').toPandas(), x='Fedu', y='avg(Dalc)')
  plt.show()
  plt.figure()
  sns.barplot(data=data.select('Dalc', 'Fedu').groupBy('Dalc', 'Fedu').count().toPandas(), x="Fedu", y='count', hue="Dalc")
  plt.show()

def question_alcohol_number_of_school_absences(data):
  correlate_class_boxplot(data, 'health', 'absences', 1, 5)

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
    __print_table_('PREDICTIONS', output)

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

def __print_table_(title, data):
    print(title)
    print(data.to_markdown())
    print()