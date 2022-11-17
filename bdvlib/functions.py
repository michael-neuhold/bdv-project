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

def get_test_value_1():
  return "Hello World - 1"

def get_test_value_2():
  return "Hello World - 2"

def question_alcohol_sex_distribution(data):
  data.filter(data.Dalc > 2).groupBy("sex").count().toPandas().plot.bar(x='sex', y='count', title='Drink a lot of alcohol during the week')
  data.filter(data.Walc > 2).groupBy("sex").count().toPandas().plot.bar(x='sex', y='count', title='Drink a lot of alcohol on the weekend') 
