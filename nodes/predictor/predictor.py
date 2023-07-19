import json
import pandas
import os
import sys
import pickle
from pathlib import Path
import time
import ast

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../utils')

#read configurations
config = json.loads(input())
print(config)
def load():
	try:
		from sklw import SKLW
		return SKLW(path=config['path'])
	except:
		try:
			from dnnctf import DNNCTF
			print(config['path'])
			return DNNCTF(path=config['path'], load=True)
		except:
			return None

while True:
  data = input()

  if not Path(config["predsPath"]).exists():
  	Path(config["predsPath"]).mkdir()    
  try:
  	# load data from request
  	features = pandas.read_json(data, orient='values')
  except:
  	# lead file specified in the request
  	data = ast.literal_eval(json.loads(data))
  	df = pandas.read_csv(data['path'], header=None)
  	features = df[[x for x in df.columns[:-1]]]
  
  model = load()
  if model is None:
  	raise('Cannot find model.')
  model.update()
  
  #pickle.dump(model.predict(features), open(Path(config["predsPath"]) / Path(config['path']).name, "wb"))
  preds = pandas.Series(model.predict(features))
  target_filename = Path(config["predsPath"]) / f"{Path(config['path']).name}.csv"
  preds.to_csv(target_filename, header=None, index=None)
  time.sleep(1)
  print({"path": str(target_filename)})
