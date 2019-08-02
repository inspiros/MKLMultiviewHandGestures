from __future__ import print_function
from collections import OrderedDict
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from MKL.algorithms import EasyMKL
from MKL.multiclass import OneVsRestMKLClassifier
from params import Params
from kernel import Kernel
from evaluate import *
import numpy as np
import datapersistant
import argparse
CONFIGURATIONS = __import__('configs')


def parse_args():
    parser = argparse.ArgumentParser(description='MultiviewHandGestures MKL Classification')
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
                        help='Kernel.', required=True)
    return parser.parse_args()


args = parse_args()

DATASET_ROOT = args.dataset_root if args.dataset_root is not None else CONFIGURATIONS.DATASET_ROOT

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

rgb_data = datapersistant.persistGestureDataset(
    os.path.join(DATASET_ROOT, MODAL_1 + '_' + KINECT_TRAIN + '_' + KINECT_TEST),
    'gesture_' + MODAL_1 + '_' + KINECT_TRAIN + '_' + KINECT_TEST, ITER_1, False)
depth_data = datapersistant.persistGestureDataset(
    os.path.join(DATASET_ROOT, MODAL_2 + '_' + KINECT_TRAIN + '_' + KINECT_TEST),
    'gesture_' + MODAL_2 + '_' + KINECT_TRAIN + '_' + KINECT_TEST, ITER_2, False)
gesture_datasets = [rgb_data, depth_data]


def leave_one_out(target, datasets):
    """
    Split data by subject
    """
    datasets = [datasets[i].singleDatasets[datasets[i].subjects.index(target)] for i in range(len(datasets))]
    Xtrs = [[] for _ in datasets]
    ytr = []
    Xtes = [[] for _ in datasets]
    yte = []

    dataDicts = []
    for i in range(len(datasets)):
        tmpDataDict = {}
        for subject in datasets[i].subjects:
            for record in subject.records:
                tmpDataDict.update({subject.name + '__' + str(record.label) + '__' + str(record.setup):
                                        {'subject': subject.name, 'setup': record.setup, 'data': record.data,
                                         'label': record.label}})
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
    print('---', 'Training on', ytr.shape[0], 'samples, testing on', yte.shape[0], 'samples')
    return Xtrs, ytr, Xtes, yte


def save_results(y_true, y_pred, verbose_string, title, subdir):
    """
    Saving results
    """
    global CONFUSION_MATRIX, CLASSIFICATION_REPORT
    if CONFUSION_MATRIX:
        # save confusion matrix
        print('[' + str(verbose_string) + ']', end=' ')
        confuse(y_true, y_pred, title=title, save=EXPORT_LEAVE_ONE_OUT, subdir=subdir)
    if CLASSIFICATION_REPORT:
        # save classification report
        report(y_true, y_pred, title=title, save=EXPORT_LEAVE_ONE_OUT, subdir=subdir)


def cross_evaluation(datasets, **kwargs):
    """
    Leave one out to test and train on others.
    """
    global GESTURE_SUBDIR
    global EXPORT_LEAVE_ONE_OUT
    global EXPORT_OVERALL
    global KINECT_TRAIN
    global KINECT_TEST
    global KERNEL_TYPE
    SAVE_SUB_DIR = os.path.join(GESTURE_SUBDIR, KINECT_TRAIN + '_' + KINECT_TEST + '_' + KERNEL_TYPE.lower())

    kernels = kwargs['kernels']
    kernel_concatenate = kwargs['kernel_concatenate']
    C_mkl = kwargs['C_mkl']
    C_svms = kwargs['C_svms']
    C_concatenate = kwargs['C_concatenate']
    lam_mkl = kwargs['lam_mkl']
    Late_fusion_weights = kwargs['late_fusion_weights']

    # Get subjects list
    subject_list = []
    for subject in datasets[0].subjects:
        for i in range(1, len(datasets)):
            if subject in datasets[i].subjects:
                subject_list.append(subject)

    y_true_overall = []
    y_true_overall_leave_one_out = [[] for _ in datasets]

    y_pred_overall_mkl = []
    y_pred_overall_svm_concate = []
    y_pred_overall_svm = [[] for _ in datasets]
    y_pred_overall_sum_late_fusion = []
    y_pred_overall_max_late_fusion = []

    # ---------------------------
    # Scores dictionary for return
    # ---------------------------
    scores = OrderedDict()
    overall_score = OrderedDict()
    # Begin leave one out evaluation
    for subject in subject_list:
        print('[LEAVE ONE OUT PROCEDURE]', 'Leaving', subject, 'out for testing')
        leave_one_out_scores = OrderedDict()
        # Split to train and test set
        Xtrs, ytr, Xtes, yte = leave_one_out(subject, datasets)

        # Compute Kernel of concatenated normalized features
        Xtrain_concate = np.concatenate([normalize(Xtr) for Xtr in Xtrs], axis=1)
        Xtest_concate = np.concatenate([normalize(Xte) for Xte in Xtes], axis=1)

        # Compute Kernels of each modality for MKL
        Ktr_concate = kernel_concatenate(Xtrain_concate)
        Kte_concate = kernel_concatenate(Xtest_concate, Xtrain_concate)

        KLtr = [None for _ in Xtrs]
        KLte = [None for _ in Xtes]
        for i in range(len(Xtrs)):
            KLtr[i] = kernels[i](Xtrs[i])
            KLte[i] = kernels[i](Xtes[i], Xtrs[i])

        # ---------------------------
        # MKL
        # ---------------------------
        base_learner = SVC(C=C_mkl, tol=0.0001, kernel='precomputed')
        clf = EasyMKL(estimator=base_learner, lam=lam_mkl, max_iter=1000, verbose=True)
        mkl = OneVsRestMKLClassifier(clf)

        # Fit and eval MKL
        mkl.fit(KLtr, ytr)
        y_pred = mkl.predict(KLte)
        mkl_score = f1(yte, y_pred)

        print()
        save_results(y_true=yte, y_pred=y_pred,
                     verbose_string='MKL',
                     title='MKL ' + Kernel.stringigfy(kernels) + ' on ' + subject,
                     subdir=os.path.join(SAVE_SUB_DIR, subject))

        y_true_overall.extend(yte)
        y_pred_overall_mkl.extend(y_pred)

        # ---------------------------
        # SVM on concatenated normalized features
        # ---------------------------
        clf_concatenate = SVC(C=C_concatenate, tol=0.0001, kernel='precomputed')
        clf_concatenate.fit(Ktr_concate, ytr)
        y_pred = clf_concatenate.predict(Kte_concate)
        svm_concate_score = f1(yte, y_pred)

        save_results(y_true=yte, y_pred=y_pred,
                     verbose_string='Early fusion',
                     title='Early fusion ' + kernel_concatenate.tostring() + ' on ' + subject,
                     subdir=os.path.join(SAVE_SUB_DIR, subject))
        y_pred_overall_svm_concate.extend(y_pred)

        # ---------------------------
        # Single modality
        # ---------------------------
        decisions = []
        for i in range(len(datasets)):
            # ---------------------------
            # SVM
            # ---------------------------
            clf = SVC(C=C_svms[i], tol=0.0001, kernel='precomputed')
            clf.fit(KLtr[i], ytr)
            y_pred = clf.predict(KLte[i])
            decisions.append(clf.decision_function(KLte[i]) * Late_fusion_weights[i])
            svm_score = f1(yte, y_pred)
            leave_one_out_scores.update({'SVM ' + kernels[i].name: svm_score})

            save_results(y_true=yte, y_pred=y_pred,
                         verbose_string='SVM on ' + kernels[i].name,
                         title='SVM ' + kernels[i].tostring() + ' on ' + subject,
                         subdir=os.path.join(SAVE_SUB_DIR, subject))

            y_true_overall_leave_one_out[i].extend(yte)
            y_pred_overall_svm[i].extend(y_pred)

        # ---------------------------
        # Sum Late fusion
        # ---------------------------
        y_pred = np.argmax(np.sum(np.array(decisions, dtype=np.float32), axis=0), axis=1).astype(np.int) + 1
        save_results(y_true=yte, y_pred=y_pred,
                     verbose_string='Sum Late fusion',
                     title='Sum Late fusion on ' + subject,
                     subdir=os.path.join(SAVE_SUB_DIR, subject))
        sum_late_fusion_score = f1(yte, y_pred)
        y_pred_overall_sum_late_fusion.extend(y_pred)

        # ---------------------------
        # Max Late fusion
        # ---------------------------
        y_pred = np.argmax(np.concatenate(decisions, axis=1), axis=1).astype(np.int) % decisions[0].shape[1] + 1
        save_results(y_true=yte, y_pred=y_pred,
                     verbose_string='Max Late fusion',
                     title='Max Late fusion on ' + subject,
                     subdir=os.path.join(SAVE_SUB_DIR, subject))
        max_late_fusion_score = f1(yte, y_pred)
        y_pred_overall_max_late_fusion.extend(y_pred)

        # Push leave-one-out scores of all subjects to dictionary for Later use
        leave_one_out_scores.update({'MKL': mkl_score,
                                     'Early fusion': svm_concate_score,
                                     'Sum Late fusion': sum_late_fusion_score,
                                     'Max Late fusion': max_late_fusion_score})
        scores.update({subject: leave_one_out_scores})
        print()

    print()
    print()
    print('SUMARIZING')
    # ---------------------------
    # Overall MKL
    # ---------------------------
    mkl_score = f1(y_true_overall, y_pred_overall_mkl)
    save_results(y_true=y_true_overall, y_pred=y_pred_overall_mkl,
                 verbose_string='Overall MKL',
                 title='MKL ' + Kernel.stringigfy(kernels),
                 subdir=SAVE_SUB_DIR)

    # ---------------------------
    # Overall Early fusion
    # ---------------------------
    svm_concate_score = f1(y_true_overall, y_pred_overall_svm_concate)
    save_results(y_true=y_true_overall, y_pred=y_pred_overall_svm_concate,
                 verbose_string='Overall Early fusion',
                 title='Early fusion ' + kernel_concatenate.tostring(),
                 subdir=SAVE_SUB_DIR)

    # ---------------------------
    # Overall Single modality
    # ---------------------------
    for i in range(len(y_pred_overall_svm)):
        # ---------------------------
        # Overall SVM
        # ---------------------------
        svm_score = f1(y_true_overall_leave_one_out[i], y_pred_overall_svm[i])
        overall_score.update({'SVM ' + kernels[i].name: svm_score})
        save_results(y_true=y_true_overall_leave_one_out[i], y_pred=y_pred_overall_svm[i],
                     verbose_string='Overall SVM on ' + kernels[i].name,
                     title='SVM ' + kernels[i].tostring(),
                     subdir=SAVE_SUB_DIR)

    # ---------------------------
    # Overall Sum Late fusion
    # ---------------------------
    sum_late_fusion_score = f1(y_true_overall, y_pred_overall_sum_late_fusion)
    save_results(y_true=y_true_overall, y_pred=y_pred_overall_sum_late_fusion,
                 verbose_string='Overall Sum Late fusion',
                 title='Sum Late fusion',
                 subdir=SAVE_SUB_DIR)

    # ---------------------------
    # Overall Max Late fusion
    # ---------------------------
    max_late_fusion_score = f1(y_true_overall, y_pred_overall_max_late_fusion)
    save_results(y_true=y_true_overall, y_pred=y_pred_overall_max_late_fusion,
                 verbose_string='Overall Max Late fusion',
                 title='Max Late fusion',
                 subdir=SAVE_SUB_DIR)

    overall_score.update({'MKL': mkl_score,
                          'Early fusion': svm_concate_score,
                          'Sum Late fusion': sum_late_fusion_score,
                          'Max Late fusion': max_late_fusion_score})
    scores.update({'Overall': overall_score})
    return scores


'''
Program begins
'''
for param in dir(CONFIGURATIONS):
    called_param = getattr(CONFIGURATIONS, str(param))
    if isinstance(called_param, Params) and called_param.is_assignable(KERNEL_TYPE):
        print('[MultiviewHandGestures MKL Classification]', called_param.name)
        print()
        result = cross_evaluation(gesture_datasets, **called_param.get_params())
        summary(result, os.path.join(GESTURE_SUBDIR, KINECT_TRAIN + '_' + KINECT_TEST + '_' + KERNEL_TYPE.lower()))
        break
