from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from sklearn.utils.multiclass import unique_labels

import os
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Patch
import numpy as np

from fileutils import assureDir
# Create directiories for containing visalizations

import warnings
warnings.filterwarnings('ignore')


RESULT_FOLDER = 'result'
assureDir(os.path.join(RESULT_FOLDER, 'confusion'))
assureDir(os.path.join(RESULT_FOLDER, 'report'))

def report(y_true, y_pred, title=None, save=False, subdir=None):
	n_classes=unique_labels(y_true, y_pred)
	reportDict = classification_report(y_true, y_pred, output_dict=True)
	keys = list(reportDict.keys())
	vals = list(reportDict.values())
	precisions = [vals[i]['precision'] for i in range(len(n_classes))]
	recalls = [vals[i]['recall'] for i in range(len(n_classes))]
	supports = [vals[i]['support'] for i in range(len(n_classes))]
	f1s = [vals[i]['f1-score'] for i in range(len(n_classes))]

	plotMat = [[precisions[i], recalls[i], supports[i], f1s[i]] for i in range(len(n_classes))]

	df_rp = pd.DataFrame(plotMat, index = [i for i in n_classes], columns = ['Precision', 'Recall', 'Support', 'F1 score'])

	if title is not None and save is True:
		if subdir is not None:
			assureDir(os.path.join(RESULT_FOLDER, 'report', subdir))
			df_rp.T.to_csv(os.path.join(RESULT_FOLDER, 'report', subdir, title + '.csv'))
		else:
			df_rp.T.to_csv(os.path.join(RESULT_FOLDER, 'report', title + '.csv'))


def f1(y_true, y_pred):
	return metrics.f1_score(y_true, y_pred, average = 'weighted')


def confuse(y_true, y_pred, weights=None, title=None, save=False, show=False, subdir=None):
	n_classes=unique_labels(y_true, y_pred)
	conf = confusion_matrix(y_true, y_pred)
	f1_score = metrics.f1_score(y_true, y_pred, average = 'weighted')
	score = "F1 Score: " + str(f1_score)
	print(score)

	if(weights is not None):
		fig = plt.subplots(1, 2, figsize=(12, 7))
		gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
		ax1 = plt.subplot(gs[0])
		ax2 = plt.subplot(gs[1])
		# fig, (ax, ax2) = plt.subplots(ncols=2, gridspec_kw = {'width_ratios':[4, 1]})
		df_cm = pd.DataFrame(conf, index = [i for i in n_classes], columns = [i for i in n_classes])
		df_w = pd.DataFrame.from_dict(weights, orient='index')
		df_w.columns = ['Kernel '+str(df_w.columns[i]) for i in df_w.columns]
		sn.heatmap(df_cm, cbar=False, annot=True, ax=ax1, fmt='d')
		if len(df_w.columns) < 5:
			sn.heatmap(df_w, cbar=False, cmap="YlGnBu", annot=True, ax=ax2, vmin=0.0, vmax = 1.0, fmt='.3g')
		else:
			sn.heatmap(df_w, cbar=False, cmap="YlGnBu", ax=ax2, vmin=0.0, vmax = 1.0)

		ax1.set_title('Confusion Matrix')
		ax1.set_xlabel('Predicted')
		ax1.set_ylabel('True Label')
		ax2.set_title('Kernel Weights')
	else:
		fig, ax = plt.subplots(1, 1, figsize=(12, 7)) 
		df_cm = pd.DataFrame(conf, index = [i for i in n_classes], columns = [i for i in n_classes])
		sn.heatmap(df_cm, cbar=False, annot=True, annot_kws={"size": 16}, fmt='d')

		ax.set_title('Confusion Matrix')
		ax.set_xlabel('Predicted')
		ax.set_ylabel('True Label')

	if title is not None:
		plt.suptitle(title + '\n' + score)
		if save is True:
			if subdir is not None:
				assureDir(os.path.join(RESULT_FOLDER, 'confusion', subdir))
				plt.savefig(os.path.join(RESULT_FOLDER, 'confusion', subdir, title + '.png'))
				df_cm.to_csv(os.path.join(RESULT_FOLDER, 'confusion', subdir, title + '.csv'))
			else:
				plt.savefig(os.path.join(RESULT_FOLDER, 'confusion', title + '.png'))
				df_cm.to_csv(os.path.join(RESULT_FOLDER, 'confusion', title + '.csv'))
	else:
		plt.suptitle(score)
	if show:
		plt.show(block=False)
		plt.pause(0.5)
	plt.close()
	return f1_score

def summary(result, directory):
	print('')
	print('Result:')
	df_scores = pd.DataFrame.from_dict(result).T
	print(df_scores)
	df_scores.to_csv(os.path.join(RESULT_FOLDER, 'confusion', directory, 'result.csv'))