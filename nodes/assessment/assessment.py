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
from string import digits 

#read configurations
config = json.loads(input())


def load_data(key):
  try:
  	# load data from request
    df = pandas.read_csv(ast.literal_eval(key)["path"], header=None)
    
    y = df[df.columns[-1]].to_numpy()
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

  scores = {} 
  
  scores["Total"] = {
      "accuracy": m.accuracy_score(y_true, y_pred),
      "f1": m.f1_score(y_true, y_pred, average='macro'),
      "precision": m.precision_score(y_true, y_pred, average='macro'),
      "recall": m.recall_score(y_true, y_pred, average='macro'),
  }

  for k, v in scores["Total"].items():
    scores["Total"][k] = np.round(v, 3)
  
  scores['Total']['conf_matrix'] = m.confusion_matrix(y_true, y_pred).tolist()

  if 'class' in config.keys() and config['class']:
    for target_class in sorted(np.unique(y_true)):
      y_true_c = [1 if y == target_class else 0 for y in y_true]
      y_pred_c = [1 if y == target_class else 0 for y in y_pred] 
  
      scores[str(target_class)] = {
        "accuracy": m.accuracy_score(y_true_c, y_pred_c),
        "f1": m.f1_score(y_true_c, y_pred_c),
        "precision": m.precision_score(y_true_c, y_pred_c),
        "recall": m.recall_score(y_true_c, y_pred_c),
        # "support": int(sum(y_true_c))
      }  
      for k, v in scores[str(target_class)].items():
        scores[str(target_class)][k] = np.round(v, 3)

  print(json.dumps(scores))
