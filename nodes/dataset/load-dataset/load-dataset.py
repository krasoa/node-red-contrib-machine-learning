import pandas
import sys
import json
from pathlib import Path

#read configurations
config = json.loads(input())

while True:
	#wait request
    input()
    if Path(config["path"]).exists():  
      if config['input'] or config['output']:
          df = pandas.read_csv(config['path'], header=None)
  
          r, c = df.shape

          if r*c > 10000:
            res = {
              "path": config["path"],
              "input": config["input"],
              "output": config["output"],
            }
          else:
            first_column = 0 if config['input'] else c - 1
            last_column = c if config['output'] else c - 1
    
            df = df.iloc[:, first_column:last_column]
    
            if config['output'] and not config['input']:
                res = json.dumps(df[df.columns[0]].tolist())
            else:
                res = df.to_json(orient='values')

          print(res)
  
      else:
          print('Nothing to load.', file=sys.stderr)
    else:
      print(f"{config['path']} does not exist. Please recheck spelling.", file=sys.stderr)
