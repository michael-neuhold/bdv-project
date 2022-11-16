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