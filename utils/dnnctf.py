#Tensorflow dnnc's wrapper
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras 
import logging
import pickle
from shutil import rmtree
import os
import numpy as np
from pathlib import Path

class DNNCTF:
	def __init__(self, path, layers = [10, 10, 10], learning_rate=0.01, steps=1000, batch_size=10, del_prev_mod=False, load=False, log=False):
		self.classifier = None
		self.last = 0
		self.del_prev_mod = del_prev_mod
		self.load = load
		if not self.__load(path):
			self.config = {
				"path": path,
				"layers": layers,
				"learning_rate": learning_rate,
				"steps": steps,
				"batch_size": batch_size
			}
		if log:
			logging.getLogger().setLevel(logging.INFO)
		else:
			logging.getLogger().setLevel(logging.ERROR)

	def __del_prev_mod(self, path):
		if path:
			try:
				rmtree(path, ignore_errors=True)
				os.makedirs(path, exist_ok=True)
			except:
				pass

	def __load(self, path):
		if self.del_prev_mod:
			self.__del_prev_mod(path)
		try:
			self.config = pickle.load(open(Path(path) / 'config.b', "rb"))
			self.classifier = keras.models.load_model(path, compile=True)
			return True
		except:
			if self.load:
				raise('Cannot load model.')
			return False

	def __input_func(self, x, y=None):
		features = {str(k):x[k] for k in x.keys()}
		if y:
			inputs = (features, y)
		else:
			inputs = features
		return tf.data.Dataset.from_tensor_slices(inputs).repeat().batch(self.config['batch_size'])

	def fit(self, x, y):
		encoder = LabelEncoder()
		y = encoder.fit_transform(y)
		self.config['classes'] = encoder.classes_
		self.config['encoder'] = encoder
      
		y = keras.utils.to_categorical(y)
		self.classifier = keras.models.Sequential()
		self.classifier.add(keras.Input(x.shape[1]))

		for layer in self.config['layers']:
			self.classifier.add(
              keras.layers.Dense(layer, activation = 'relu')
            )
		self.classifier.add(keras.layers.Dense(y.shape[1], activation='sigmoid'))
		self.classifier.compile(loss="categorical_crossentropy", 
                                optimizer=keras.optimizers.SGD(learning_rate=self.config['learning_rate']), 
                                metrics=["accuracy"])

		self.classifier.fit(x,y,batch_size=self.config['batch_size'], epochs=self.config['steps'], verbose=2)
		self.__save()
	

	def predict(self, x):
		preds = self.classifier.predict(x).argmax(axis=1)
		if self.config['encoder'] is not None: 
			preds = self.config['encoder'].inverse_transform(preds)
		else:
			preds = np.array([self.config['classes'][i] for i in preds])
		return preds

	def __save(self):
		self.classifier.save(self.config['path'])
		pickle.dump(self.config, open(Path(self.config['path']) / 'config.b', "wb"))

	def update(self):
		try:
			modified = os.stat(Path(self.config['path']) / 'config.b').st_mtime
		except:
			modified = 0
		if(modified > self.last):
			self.last = modified
			self.__load(self.config['path'])
