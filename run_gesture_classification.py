import os
import sys
import datapersistant
import numpy as np
import pandas as pd
import argparse
from kernel import Kernel
from sklearn.preprocessing import normalize

from MKL.algorithms import EasyMKL
from MKL.multiclass import OneVsRestMKLClassifier

from configs import CONFIGS
from sklearn.svm import SVC

from evaluate import *

def parse_args():
	parser = argparse.ArgumentParser(description='MKL')
	parser.add_argument('--kinect_train', type=str, default=None,
	                        help='Kinect train.')
	parser.add_argument('--kinect_test', type=str, default=None,
	                        help='Kinect test.')
	parser.add_argument('--iter_rgb', type=int, default=None,
	                        help='Iteration of RGB features.')
	parser.add_argument('--iter_depth', type=int, default=None,
	                        help='Iteration of Depth features.')
	parser.add_argument('--dataset_root', type=str, default=None,
	                        help='Dataset Root.')

	parser.add_argument('--export_leave_one_out', type=bool, default=True,
	                        help='Dataset Root.')
	parser.add_argument('--export_overall', type=bool, default=True,
	                        help='Dataset Root.')
	parser.add_argument('--confusion_matrix', type=bool, default=True,
	                        help='Confusion Matrix.')
	parser.add_argument('--classification_report', type=bool, default=True,
	                        help='Classification Report.')

	parser.add_argument('--rgb', type=str, default='RGB',
	                        help='RGB folder.')
	parser.add_argument('--depth', type=str, default='Depth',
	                        help='Depth folder.')

	parser.add_argument('--kernels', type=str, default=None,
	                        help='Kernel.')
	return parser.parse_args()

args = parse_args()

DATASET_ROOT = args.dataset_root if args.dataset_root is not None else DATASET_ROOT

EXPORT_LEAVE_ONE_OUT = args.export_leave_one_out
EXPORT_OVERALL = args.export_overall
CONFUSION_MATRIX = args.confusion_matrix
CLASSIFICATION_REPORT = args.classification_report

KINECT_TRAIN = args.kinect_train
KINECT_TEST = args.kinect_test

KERNEL_TYPE = args.kernels

MODAL_1 = args.rgb
MODAL_2 = args.depth

ITER_1 = args.iter_rgb
ITER_2 = args.iter_depth

GESTURE_SUBDIR = os.path.join(MODAL_1 + '_' + str(ITER_1) + ' ' + MODAL_2 + '_' + str(ITER_2))

rgb_data = datapersistant.persistGestureDataset(os.path.join(DATASET_ROOT, MODAL_1 + '_' + KINECT_TRAIN + '_' + KINECT_TEST), 'gesture_' + MODAL_1 + '_' + KINECT_TRAIN + '_' + KINECT_TEST, ITER_1, False)
depth_data = datapersistant.persistGestureDataset(os.path.join(DATASET_ROOT, MODAL_2 + '_' + KINECT_TRAIN + '_' + KINECT_TEST), 'gesture_' + MODAL_2 + '_' + KINECT_TRAIN + '_' + KINECT_TEST, ITER_2, False)
gesture_datasets = [rgb_data, depth_data]



# SPLIT DATA BY SUBJECT
def leave_one_out(target, datasets):
	datasets = [datasets[i].singleDatasets[datasets[i].subjects.index(target)] for i in range(len(datasets))]
	Xtrs = [[] for i in datasets]
	ytr = []
	Xtes = [[] for i in datasets]
	yte = []

	dataDicts = []
	for i in range(len(datasets)):
		tmpDataDict = {}
		for subject in datasets[i].subjects:
			for record in subject.records:
				tmpDataDict.update({subject.name + '__' + str(record.label) + '__' + str(record.setup) : 
					{'subject':subject.name, 'setup':record.setup, 'data':record.data, 'label':record.label}})
		dataDicts.append(tmpDataDict)

	for key in dataDicts[0].keys():
		present_in_all = True
		for j in range(1, len(dataDicts)):
			if key not in dataDicts[j].keys():
				present_in_all = False

		if present_in_all:
			for i in range(len(dataDicts)):
				if key.rfind(target + '__') != 0:
					Xtrs[i].append(dataDicts[i][key]['data'])
					if i == 0:
						ytr.append(dataDicts[i][key]['label'])
				else:
					Xtes[i].append(dataDicts[i][key]['data'])
					if i == 0:
						yte.append(dataDicts[i][key]['label'])

	for i in range(len(Xtrs)):
		Xtrs[i] = np.array(Xtrs[i])
		Xtes[i] = np.array(Xtes[i])
	ytr = np.array(ytr)
	yte = np.array(yte)
	print(target, ytr.shape[0], yte.shape[0])
	return Xtrs, ytr, Xtes, yte


def cross_evaluation(datasets, kernels, kernels_concate=None, C_mkl=None, C_svms=None, C_concatenate=None, lam_mkl=None):
	'''
	Leave one out to test and train on others.
	'''
	global GESTURE_SUBDIR
	global EXPORT_LEAVE_ONE_OUT
	global EXPORT_OVERALL
	global KINECT_TRAIN
	global KINECT_TEST
	global KERNEL_TYPE

	global CONFUSION_MATRIX
	global CLASSIFICATION_REPORT
	SAVE_SUB_DIR = os.path.join(GESTURE_SUBDIR, KINECT_TRAIN + '_' + KINECT_TEST + '_' + KERNEL_TYPE.lower())

	# Get subjects list
	subject_list = []
	for subject in datasets[0].subjects:
		for i in range(1, len(datasets)):
			if subject in datasets[i].subjects:
				subject_list.append(subject)

	# Begin cross evaluation
	y_true_overall_mkl = []
	y_pred_overall_mkl = []
	y_true_overall_svm_concate = []
	y_pred_overall_svm_concate = []
	y_true_overall_svm = [[] for i in datasets]
	y_pred_overall_svm = [[] for i in datasets]
	scores = {}
	for subject in subject_list:
		print('Leaving', subject, 'out for testing')
		# Split to train and test set
		Xtrs, ytr, Xtes, yte = leave_one_out(subject, datasets)

		# Compute Kernel of concatenated normalized features
		Xtrain_concate = np.concatenate([normalize(Xtr) for Xtr in Xtrs], axis=1)
		Xtest_concate = np.concatenate([normalize(Xte) for Xte in Xtes], axis=1)

		# Compute Kernels of each modality for MKL
		Ktr_concate = kernels_concate.apply(Xtrain_concate)
		Kte_concate = kernels_concate.apply(Xtest_concate, Xtrain_concate)

		KLtr = [None for i in Xtrs]
		KLte = [None for i in Xtes]
		loo_scores = {}
		for i in range(len(Xtrs)):
			KLtr[i] = kernels[i].apply(Xtrs[i])
			KLte[i] = kernels[i].apply(Xtes[i], Xtrs[i])

		# Initialize MKL
		base_learner = SVC(C=C_mkl, tol=0.0001, kernel='precomputed')
		clf = EasyMKL(estimator=base_learner, lam=lam_mkl, max_iter=1000, verbose=True)
		mkl = OneVsRestMKLClassifier(clf)

		# Fit and eval MKL
		mkl.fit(KLtr, ytr)
		y_pred = mkl.predict(KLte)
		mkl_score = f1(yte, y_pred)

		if CONFUSION_MATRIX:
			# save confusion matrix of MKL
			print('MKL')
			confuse(yte, y_pred, weights=mkl.weights, title='EasyMKL ' + Kernel.stringigfy(kernels) + ' on ' + subject, save=EXPORT_LEAVE_ONE_OUT, subdir=os.path.join(SAVE_SUB_DIR, subject))
		if CLASSIFICATION_REPORT:
			# save classification report of MKL
			report(yte, y_pred, title='EasyMKL ' + Kernel.stringigfy(kernels) + ' on ' + subject, save=EXPORT_LEAVE_ONE_OUT, subdir=os.path.join(SAVE_SUB_DIR, subject))

		y_true_overall_mkl.extend(yte)
		y_pred_overall_mkl.extend(y_pred)

		# Initialize, fit and eval SVM on concatenated features
		clf_concatenate = SVC(C=C_concatenate, tol=0.0001, kernel='precomputed')
		clf_concatenate.fit(Ktr_concate, ytr)
		y_pred = clf_concatenate.predict(Kte_concate)
		svm_concate_score = f1(yte, y_pred)

		if CONFUSION_MATRIX:
			# save confusion matrix of SVM on concatenated features
			print('SVM on concatenated features')
			confuse(yte, y_pred, title='SVM ' + kernels_concate.tostring() + ' on ' + subject, save=EXPORT_LEAVE_ONE_OUT, subdir=os.path.join(SAVE_SUB_DIR, subject))
		if CLASSIFICATION_REPORT:
			# save classification report of SVM on concatenated features
			report(yte, y_pred, title='SVM ' + kernels_concate.tostring() + ' on ' + subject, save=EXPORT_LEAVE_ONE_OUT, subdir=os.path.join(SAVE_SUB_DIR, subject))

		y_true_overall_svm_concate.extend(yte)
		y_pred_overall_svm_concate.extend(y_pred)

		'''
		SVM on each modality
		'''
		for i in range(len(datasets)):
			# Fit and eval SVM on single modality
			clf = SVC(C=C_svms[i], tol=0.0001, kernel='precomputed')
			clf.fit(KLtr[i], ytr)
			y_pred = clf.predict(KLte[i])
			svm_score = f1(yte, y_pred)
			loo_scores.update({'SVM ' + kernels[i].name:svm_score})
			if CONFUSION_MATRIX:
				# save confusion matrix of SVM on single modality
				print('SVM on ' + kernels[i].name)
				confuse(yte, y_pred, title='SVM ' + kernels[i].tostring() + ' on ' + subject, save=EXPORT_LEAVE_ONE_OUT, subdir=os.path.join(SAVE_SUB_DIR, subject))
			if CLASSIFICATION_REPORT:
				# save classification report of SVM on single modality
				report(yte, y_pred, title='SVM ' + kernels[i].tostring() + ' on ' + subject, save=EXPORT_LEAVE_ONE_OUT, subdir=os.path.join(SAVE_SUB_DIR, subject))

			y_true_overall_svm[i].extend(yte)
			y_pred_overall_svm[i].extend(y_pred)

		# Push leave-one-out scores of all subjects to dictionary for later use
		loo_scores.update({'EasyMKL':mkl_score, 'SVM':svm_concate_score})
		scores.update({subject:loo_scores})
		print('')

	print('')
	print('SUMARIZING')
	# The overall score - combination of leave-one-out scores
	overall_score = {}
	mkl_score = f1(y_true_overall_mkl, y_pred_overall_mkl)
	svm_concate_score = f1(y_true_overall_svm_concate, y_pred_overall_svm_concate)
	if CONFUSION_MATRIX:
		# save overall confusion matrix of MKL and SVM on concatenated features
		print('Overall MKL')
		confuse(y_true_overall_mkl, y_pred_overall_mkl, title='EasyMKL ' + Kernel.stringigfy(kernels), save=EXPORT_OVERALL, subdir=SAVE_SUB_DIR)
		print('Overall SVM on concatenated features')
		confuse(y_true_overall_svm_concate, y_pred_overall_svm_concate, title='SVM ' + kernels_concate.tostring(), save=EXPORT_OVERALL, subdir=SAVE_SUB_DIR)
	if CLASSIFICATION_REPORT:
		# save overall classification report of MKL and SVM on concatenated features
		report(y_true_overall_mkl, y_pred_overall_mkl, title='EasyMKL ' + Kernel.stringigfy(kernels), save=EXPORT_OVERALL, subdir=SAVE_SUB_DIR)
		report(y_true_overall_svm_concate, y_pred_overall_svm_concate, title='SVM ' + kernels_concate.tostring(), save=EXPORT_OVERALL, subdir=SAVE_SUB_DIR)
	for i in range(len(y_true_overall_svm)):
		svm_score = f1(y_true_overall_svm[i], y_pred_overall_svm[i])
		overall_score.update({'SVM ' + kernels[i].name:svm_score})
		if CONFUSION_MATRIX:
			# save overall confusion matrix of SVM on each subject
			print('Overall SVM on ' + kernels[i].name)
			confuse(y_true_overall_svm[i], y_pred_overall_svm[i], title='SVM ' + kernels[i].tostring(), save=EXPORT_OVERALL, subdir=SAVE_SUB_DIR)
		if CLASSIFICATION_REPORT:
			# save overall classification report of SVM on each subject
			report(y_true_overall_svm[i], y_pred_overall_svm[i], title='SVM ' + kernels[i].tostring(), save=EXPORT_OVERALL, subdir=SAVE_SUB_DIR)
	overall_score.update({'EasyMKL':mkl_score})
	overall_score.update({'SVM':svm_concate_score})
	scores.update({'Overall':overall_score})
	return scores




'''
Program begins
'''
for config in CONFIGS:
	if config.is_assignable(KERNEL_TYPE):
		print('[MKL Classification Begins]', config.name)
		result = cross_evaluation(gesture_datasets, *config.to_params())
		summary(result, os.path.join(GESTURE_SUBDIR, KINECT_TRAIN + '_' + KINECT_TEST + '_' + KERNEL_TYPE.lower()))
		break
