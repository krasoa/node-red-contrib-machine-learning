import json
import pickle
import pandas
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../utils')
from sklw import SKLW
import ast


#read configurations
config = json.loads(input())
save = config['save']

while True:
#read request
  data = input()
  try:
  	# load data from request
  	df = pandas.read_json(data, orient='values')
  except:
  	# lead file specified in the request
  	data = ast.literal_eval(json.loads(data))
  	df = pandas.read_csv(data['path'], header=None)
# df = pandas.read_csv("/data/datasets/mnist/train.csv", header=None)

  x = df.iloc[:, :-1]
  y = df.iloc[:, -1].to_numpy().flatten()

  classifier = None

  if config['classifier'] == 'decision-tree-classifier':
  	from sklearn.tree import DecisionTreeClassifier
  	classifier = SKLW(path=save, model=DecisionTreeClassifier(**config['kwargs']))
  elif config['classifier'] == 'deep-neural-network-classifier-tensorflow':
  	from dnnctf import DNNCTF
  	classifier = DNNCTF(path=save, del_prev_mod=True, **config['kwargs'])
  elif config['classifier'] == 'random-forest-classifier':
  	from sklearn.ensemble import RandomForestClassifier
  	classifier = SKLW(path=save, model=RandomForestClassifier(**config['kwargs']))

  try:
  	#train model
  	classifier.fit(x, y)
  except Exception as e:
  	print(e)
  	raise()

  print(config['classifier'] + ': training completed.')
