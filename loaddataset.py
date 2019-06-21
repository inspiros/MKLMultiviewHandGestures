import gestureconfig
import blob
import numpy as np
import pickle
import re
import os

class Record:
	def __init__(self, data = None, label = None, setup = None):
		self.data = data
		self.label = label
		self.setup = setup

class Subject:
	def __init__(self, id, name = None):
		self.id = id
		self.name = name
		self.records = list()

class SingleDataset:
	def __init__(self):
		self.subjects = list()

class MultiviewDataset:
	def __init__(self):
		self.singleDatasets = list()
		self.subjects = list()


# Loaders
class GestureDatasetLoader:
	def __init__(self, directory, iteration):
		self.dataset = MultiviewDataset()

		for testsjdir in os.listdir(directory):
			test_subject_dir = os.path.join(directory, testsjdir, 'feature', 'iter_' + str(iteration))
			dataset = SingleDataset()
			for sjdir in os.listdir(test_subject_dir):
				subject_dir = os.path.join(test_subject_dir, sjdir)

				subject = Subject(len(self.dataset.subjects) + 1, sjdir)
				for cldir in os.listdir(subject_dir):
					classdir = os.path.join(subject_dir, cldir)
					for sudir in os.listdir(classdir):
						setupdir = os.path.join(classdir, sudir)
						fc_file = os.path.join(setupdir, '000001.' + gestureconfig.layer)
						data = blob.load_np_array(fc_file)

						rc = Record()
						rc.data = data
						rc.label = int(cldir)
						rc.setup = int(sudir)
						subject.records.append(rc)
						print(sjdir, cldir, sudir)
				dataset.subjects.append(subject)
			self.dataset.singleDatasets.append(dataset)
			self.dataset.subjects.append(testsjdir)
