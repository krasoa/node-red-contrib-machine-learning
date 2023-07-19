import sklearn.metrics as m
import json
import sys
import numpy as np
from inspect import getargspec
import pickle
import ast
from pathlib import Path
import pandas
from pprint import pprint

#read configurations
config = json.loads(input())


def load_data(key):
  try:
  	# load data from request
    df = pandas.read_csv(ast.literal_eval(key)["path"], header=None)
    
    y = df[df.columns[-1]]
  except Exception as e:
  	# lead file specified in the request
    y = np.array(key)
  return y

while True:
  #read request
  data = json.loads(input())
  real = data["real"]
  pred = data["predicted"]

  y_true = load_data(real)
  y_pred = load_data(pred)

  scores = {
  	"accuracy": m.accuracy_score(y_true, y_pred),
  	"f1": m.f1_score(y_true, y_pred, average='micro'),
  	"precision": m.precision_score(y_true, y_pred, average='micro'),
  	"recall": m.recall_score(y_true, y_pred, average='micro'),
  }
  for k, v in scores.items():
    scores[k] = np.round(v, 3)
  pprint(scores)
