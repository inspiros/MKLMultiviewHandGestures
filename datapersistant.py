import pickle
import os
from loaddataset import *

PKL_DIRECTORY = 'temp'
from fileutils import assureDir
# Create directiories for containing visalizations
assureDir(PKL_DIRECTORY)

def save(obj, file):
	with open(os.path.join(PKL_DIRECTORY, file), 'wb') as output:
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load(file):
	if not os.path.isfile(os.path.join(PKL_DIRECTORY, file)):
		return None
	with open(os.path.join(PKL_DIRECTORY, file), 'rb') as input:
		obj = pickle.load(input)
		return obj

def loadPython2(file):
	if not os.path.isfile(file):
		return None
	with open(file, 'rb') as input:
		u = pickle._Unpickler(input)
		u.encoding = 'latin1'
		p = u.load()
		return p

def remove(file):
	if os.path.isfile(os.path.join(PKL_DIRECTORY, file)):
		os.remove(os.path.join(PKL_DIRECTORY, file))

def persistGestureDataset(origin, pkl, iteration, overwrite=False):
	if overwrite or load(pkl + '__' + str(iteration) + '.pkl') is None:
		dataset = GestureDatasetLoader(origin, iteration).dataset
		print('--- Persisting', pkl + '__' + str(iteration) + '.pkl', '---')
		save(dataset, pkl + '__' + str(iteration) + '.pkl')
		return dataset
	return load(pkl + '__' + str(iteration) + '.pkl')
