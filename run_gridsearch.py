from __future__ import print_function
from collections import OrderedDict
from sklearn.metrics import pairwise
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from MKL.algorithms import EasyMKL
from MKL.multiclass import OneVsRestMKLClassifier
from gridsearch_params import GridSearchParams
from evaluate import *
import numpy as np
import datapersistant
import argparse
import json

CONFIGURATIONS = __import__('gridsearch_configs')


def parse_args():
    parser = argparse.ArgumentParser(description='MultiviewHandGestures MKL Classification')
    parser.add_argument('--kinect', type=str, default=None,
                        help='Kinect train.')
    parser.add_argument('--iter_rgb', type=int, default=None,
                        help='Iteration of RGB features.')
    parser.add_argument('--iter_depth', type=int, default=None,
                        help='Iteration of Depth features.')
    parser.add_argument('--dataset_root', type=str, default=None,
                        help='Dataset Root.')

    parser.add_argument('--rgb', type=str, default='RGB',
                        help='RGB folder.')
    parser.add_argument('--depth', type=str, default='Depth',
                        help='Depth folder.')

    parser.add_argument('--kernels', type=str, default=None,
                        help='Kernel.', required=True)
    return parser.parse_args()


args = parse_args()

DATASET_ROOT = args.dataset_root if args.dataset_root is not None else CONFIGURATIONS.DATASET_ROOT

KINECT_TRAIN = args.kinect
KINECT_TEST = KINECT_TRAIN

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


def get_MKL(C_mkl, lam_mkl):
    base_learner = SVC(C=C_mkl, tol=0.0001, kernel='precomputed')
    clf = EasyMKL(estimator=base_learner, lam=lam_mkl, max_iter=1000, verbose=False)
    return OneVsRestMKLClassifier(clf, verbose=False)


def get_SVM(C_svm):
    return OneVsRestClassifier(SVC(C=C_svm, tol=0.0001, kernel='precomputed'), n_jobs=-1)


def evaluate_classifiers(datasets, **kwargs):
    subject_list = kwargs['subject_list']

    scores = OrderedDict()
    leave_one_out_trues = []
    leave_one_out_preds = OrderedDict()
    leave_one_out_preds['mkl'] = []
    leave_one_out_preds['svm_concatenate'] = []
    for subject in subject_list:
        print('[LEAVE ONE OUT PROCEDURE]', 'Leaving', subject, 'out for testing')
        # Split to train and test set
        Xtrs, ytr, Xtes, yte = leave_one_out(subject, datasets)
        if 'svms' not in leave_one_out_preds.keys():
            leave_one_out_preds['svms'] = [[] for _ in Xtrs]
        if 'late_fuse' not in leave_one_out_preds.keys():
            leave_one_out_preds['late_fuse'] = [[] for _ in Xtrs]
        leave_one_out_trues.append(yte)

        if 'mkl' in kwargs['clf'] or 'svms' in kwargs['clf'] or 'late_fuse_sum' in kwargs['clf'] or 'late_fuse_max' in \
                kwargs['clf']:
            kernels = kwargs['kernels']
            KLtr = [None for _ in Xtrs]
            KLte = [None for _ in Xtes]
            for i in range(len(Xtrs)):
                KLtr[i] = kernels[i](Xtrs[i])
                KLte[i] = kernels[i](Xtes[i], Xtrs[i])

            if 'mkl' in kwargs['clf']:
                mkl = get_MKL(kwargs['C_mkl'], kwargs['lam_mkl'])
                mkl.fit(KLtr, ytr)
                y_pred = mkl.predict(KLte)
                leave_one_out_preds['mkl'].append(y_pred)

        if 'svm_concatenate' in kwargs['clf']:
            # Compute Kernel of concatenated normalized features
            Xtrain_concate = np.concatenate([normalize(Xtr) for Xtr in Xtrs], axis=1)
            Xtest_concate = np.concatenate([normalize(Xte) for Xte in Xtes], axis=1)
            kernel_concatenate = kwargs['kernel_concatenate']
            Ktr_concate = kernel_concatenate(Xtrain_concate)
            Kte_concate = kernel_concatenate(Xtest_concate, Xtrain_concate)

            svm = get_SVM(kwargs['C_svm_concatenate'])
            svm.fit(Ktr_concate, ytr)
            y_pred = svm.predict(Kte_concate)
            leave_one_out_preds['svm_concatenate'].append(y_pred)

        if 'svms' in kwargs['clf'] or 'late_fuse_sum' in kwargs['clf'] or 'late_fuse_max' in kwargs['clf']:
            for i in range(len(Xtrs)):
                svm = get_SVM(kwargs['C_svms'][i])
                svm.fit(KLtr[i], ytr)
                y_pred = svm.predict(KLte[i])
                leave_one_out_preds['svms'][i].append(y_pred)

                if 'late_fuse_sum' in kwargs['clf'] or 'late_fuse_max' in kwargs['clf']:
                    late_fuse_weights = kwargs['late_fuse_weights']
                    decision = svm.decision_function(KLte[i]) * late_fuse_weights[i]
                    leave_one_out_preds['late_fuse'][i].append(decision)

    leave_one_out_trues = np.concatenate(leave_one_out_trues, axis=0)
    for key, val in leave_one_out_preds.items():
        if key in kwargs['clf'] or key == 'late_fuse':
            if key == 'mkl':
                val = np.concatenate(val, axis=0)
                scores[key] = {'C': kwargs['C_mkl'], 'lam': kwargs['lam_mkl'],
                               'f1': np.sum(f1(leave_one_out_trues, val))}
            elif key == 'svm_concatenate':
                val = np.concatenate(val, axis=0)
                scores[key] = {'C': kwargs['C_svm_concatenate'],
                               'f1': np.sum(f1(leave_one_out_trues, val))}
            elif key == 'svms':
                val = [np.concatenate(preds, axis=0) for preds in val]
                scores[key] = {'C_svms': kwargs['C_svms'],
                               'f1': [np.sum(f1(leave_one_out_trues, preds)) for preds in val]}
            elif key == 'late_fuse' and 'late_fuse_sum' in kwargs['clf'] or 'late_fuse_max' in kwargs['clf']:
                val = [np.concatenate(decisions, axis=0) for decisions in val]
                if 'late_fuse_sum' in kwargs['clf']:
                    val = np.argmax(np.sum(val, axis=0), axis=1).astype(np.int) + 1
                    scores['late_fuse_sum'] = {'late_fuse_weights': kwargs['late_fuse_weights'],
                                               'f1': np.sum(f1(leave_one_out_trues, val))}
                if 'late_fuse_max' in kwargs['clf']:
                    val = np.argmax(np.concatenate(val, axis=1), axis=1).astype(np.int) % val[0].shape[1] + 1
                    scores['late_fuse_max'] = {'late_fuse_weights': kwargs['late_fuse_weights'],
                                               'f1': np.sum(f1(leave_one_out_trues, val))}
            else:
                pass
    print()
    return scores


def grid_search_kernels(datasets, subject_list, kernel_funcs):
    if len(kernel_funcs) > 1:
        best_kernels_scores = {'rgb': 0, 'depth': 0}
        best_kernels = {'rgb': 0, 'depth': 0}
        for i_kernel_rgb in range(len(kernel_funcs)):
            score = evaluate_classifiers(datasets,
                                         subject_list=subject_list,
                                         kernels=[kernel_funcs[i_kernel_rgb], kernel_funcs[int(len(kernel_funcs) / 2)]],
                                         C_mkl=0.1,
                                         lam_mkl=None,
                                         clf=['mkl']
                                         )['mkl']['f1']
            if score >= best_kernels_scores['rgb']:
                print('[NEW]', score)
                best_kernels_scores['rgb'] = score
                best_kernels['rgb'] = i_kernel_rgb
            else:
                break
        for i_kernel_depth in range(len(kernel_funcs)):
            score = evaluate_classifiers(datasets,
                                         subject_list=subject_list,
                                         kernels=[kernel_funcs[best_kernels['rgb']], kernel_funcs[i_kernel_depth]],
                                         C_mkl=0.1,
                                         lam_mkl=None,
                                         clf=['mkl']
                                         )['mkl']['f1']
            if score >= best_kernels_scores['depth']:
                print('[NEW]', score)
                best_kernels_scores['depth'] = score
                best_kernels['depth'] = i_kernel_depth
            else:
                break
        return best_kernels_scores, best_kernels['rgb'], best_kernels['depth']
    else:
        return np.NaN, 0, 0


def grid_search_mkl(datasets, subject_list, kernels, Cs, lams):
    best_scores = {'C': 0, 'lam': 0}
    best_mkl = {'C': 0, 'lam': 0}
    for i_C_mkl in range(len(Cs)):
        score = evaluate_classifiers(datasets,
                                     subject_list=subject_list,
                                     kernels=[kernels[0], kernels[1]],
                                     C_mkl=Cs[i_C_mkl],
                                     lam_mkl=None,
                                     clf=['mkl']
                                     )['mkl']['f1']
        if score > best_scores['C']:
            print('[NEW]', score)
            best_scores['C'] = score
            best_mkl['C'] = i_C_mkl
        elif score == best_scores['C'] and i_C_mkl > len(Cs) / 2:
            break
    for i_lam_mkl in range(len(lams)):
        score = evaluate_classifiers(datasets,
                                     subject_list=subject_list,
                                     kernels=[kernels[0], kernels[1]],
                                     C_mkl=Cs[best_mkl['C']],
                                     lam_mkl=lams[i_lam_mkl],
                                     clf=['mkl']
                                     )['mkl']['f1']
        if score >= best_scores['lam']:
            print('[NEW]', score)
            best_scores['lam'] = score
            best_mkl['lam'] = i_lam_mkl
    return best_scores, best_mkl['C'], best_mkl['lam']


def grid_search_svm_concatenate(datasets, subject_list, kernel_funcs, Cs):
    best_scores = {'kernel': 0, 'C': 0}
    best_svm_concatenate = {'kernel': 0, 'C': 0}
    for i_kernel_concatenate in range(len(kernel_funcs)):
        score = evaluate_classifiers(datasets,
                                     subject_list=subject_list,
                                     kernel_concatenate=kernel_funcs[i_kernel_concatenate],
                                     C_svm_concatenate=0.1,
                                     clf=['svm_concatenate']
                                     )['svm_concatenate']['f1']
        if score >= best_scores['kernel']:
            print('[NEW]', score)
            best_scores['kernel'] = score
            best_svm_concatenate['kernel'] = i_kernel_concatenate
        else:
            break
    for i_C_svm_concatenate in range(len(Cs)):
        score = evaluate_classifiers(datasets,
                                     subject_list=subject_list,
                                     kernel_concatenate=kernel_funcs[best_svm_concatenate['kernel']],
                                     C_svm_concatenate=Cs[i_C_svm_concatenate],
                                     clf=['svm_concatenate']
                                     )['svm_concatenate']['f1']
        if score > best_scores['C']:
            print('[NEW]', score)
            best_scores['C'] = score
            best_svm_concatenate['C'] = i_C_svm_concatenate
        elif score == best_scores['C'] and i_C_svm_concatenate > len(Cs) / 2:
            break
    return best_scores, best_svm_concatenate['kernel'], best_svm_concatenate['C']


def grid_search_svms(datasets, subject_list, kernel_funcs, Cs):
    best_scores = {'C_rgb': 0, 'C_depth': 0}
    best_svms = {'C_rgb': 0, 'C_depth': 0}
    for i_C_svm_rgb in range(len(Cs)):
        score = evaluate_classifiers(datasets,
                                     subject_list=subject_list,
                                     kernels=[kernel_funcs[0], kernel_funcs[1]],
                                     C_svms=[Cs[i_C_svm_rgb], 0.1],
                                     clf=['svms']
                                     )['svms']['f1'][0]
        if score > best_scores['C_rgb']:
            print('[NEW]', score)
            best_scores['C_rgb'] = score
            best_svms['C_rgb'] = i_C_svm_rgb
        elif score == best_scores['C_rgb'] and i_C_svm_rgb > len(Cs) / 2:
            break
    for i_C_svm_depth in range(len(Cs)):
        score = evaluate_classifiers(datasets,
                                     subject_list=subject_list,
                                     kernels=[kernel_funcs[0], kernel_funcs[1]],
                                     C_svms=[Cs[best_svms['C_rgb']], Cs[i_C_svm_depth]],
                                     clf=['svms']
                                     )['svms']['f1'][1]
        if score > best_scores['C_depth']:
            print('[NEW]', score)
            best_scores['C_depth'] = score
            best_svms['C_depth'] = i_C_svm_depth
        elif score == best_scores['C_depth'] and i_C_svm_depth > len(Cs) / 2:
            break
    return best_scores, best_svms['C_rgb'], best_svms['C_depth']


def random_search_late_fuse(datasets, subject_list, kernel_funcs, Cs, max_iter=25):
    best_scores = {'w_sum': 0, 'w_max': 0}
    best_weights = {'w_sum': [], 'w_max': []}
    rgb_weight_random = lambda: np.random.uniform(.5, 1)
    for i in range(max_iter):
        w_rgb_sum = rgb_weight_random()
        w_depth_sum = 1 - w_rgb_sum
        score = evaluate_classifiers(datasets,
                                     subject_list=subject_list,
                                     kernels=[kernel_funcs[0], kernel_funcs[1]],
                                     C_svms=[Cs[0], Cs[1]],
                                     late_fuse_weights=[w_rgb_sum, w_depth_sum],
                                     clf=['late_fuse_sum']
                                     )['late_fuse_sum']['f1']
        if score > best_scores['w_sum']:
            print('[NEW]', score)
            best_scores['w_sum'] = score
            best_weights['w_sum'] = [w_rgb_sum, w_depth_sum]
    for i in range(max_iter):
        w_rgb_max = rgb_weight_random()
        w_depth_max = 1 - w_rgb_max
        score = evaluate_classifiers(datasets,
                                     subject_list=subject_list,
                                     kernels=[kernel_funcs[0], kernel_funcs[1]],
                                     C_svms=[Cs[0], Cs[1]],
                                     late_fuse_weights=[w_rgb_max, w_depth_max],
                                     clf=['late_fuse_max']
                                     )['late_fuse_max']['f1']
        if score > best_scores['w_max']:
            print('[NEW]', score)
            best_scores['w_max'] = score
            best_weights['w_max'] = [w_rgb_max, w_depth_max]
    return best_scores, best_weights['w_sum'], best_weights['w_max']


def start_grid_search(datasets, **kwargs):
    global KINECT_TRAIN
    subject_list = []
    for should_add, subject in enumerate(datasets[0].subjects):
        for i in range(1, len(datasets)):
            if subject in datasets[i].subjects:
                subject_list.append(subject)

    kernel_funcs = kwargs['kernel_funcs']
    Cs = kwargs['Cs']
    lams_mkl = kwargs['lams_mkl']

    # -----------------
    # MKL + kernels
    # -----------------
    print('MKL + Kernels')
    _, i_kernel_rgb, i_kernel_depth = grid_search_kernels(datasets, subject_list, kernel_funcs)
    _, i_C_mkl, i_lam_mkl = grid_search_mkl(datasets, subject_list,
                                            [kernel_funcs[i_kernel_rgb], kernel_funcs[i_kernel_depth]], Cs, lams_mkl)
    mkl_result = {'i_kernel_rgb': i_kernel_rgb, 'i_kernel_depth': i_kernel_depth, 'i_C_mkl': i_C_mkl,
                  'i_lam_mkl': i_lam_mkl}
    mkl_result = json.dumps(mkl_result)
    with open(KINECT_TRAIN + '_' + KERNEL_TYPE + '_mkl.json', 'w') as f:
        print('MKL:', mkl_result)
        f.write(mkl_result)
        print()
        print()
        print()

    # -----------------
    # Early fusion
    # -----------------
    print('Early fusion')
    _, i_kernel_concatenate, i_C_concatenate = grid_search_svm_concatenate(datasets, subject_list, kernel_funcs, Cs)
    svm_concatenate_result = {'i_kernel_concatenate': i_kernel_concatenate, 'i_C_concatenate': i_C_concatenate}
    svm_concatenate_result = json.dumps(svm_concatenate_result)
    with open(KINECT_TRAIN + '_' + KERNEL_TYPE + '_svm_concatenate.json', 'w') as f:
        print('SVM concatenate:', svm_concatenate_result)
        f.write(svm_concatenate_result)
        print()
        print()
        print()

    # -----------------
    # SVMs
    # -----------------
    print('SVMs')
    _, i_C_svm_rgb, i_C_svm_depth = grid_search_svms(datasets, subject_list,
                                                     [kernel_funcs[i_kernel_rgb], kernel_funcs[i_kernel_depth]], Cs)
    svms_result = {'i_C_svm_rgb': i_C_svm_rgb, 'i_C_svm_depth': i_C_svm_depth}
    svms_result = json.dumps(svms_result)
    with open(KINECT_TRAIN + '_' + KERNEL_TYPE + '_svms.json', 'w') as f:
        print('SVMs:', svms_result)
        f.write(svms_result)
        print()
        print()
        print()

    # -----------------
    # Late fusions
    # -----------------
    print('Late fusions')
    _, weights_sum, weights_max = random_search_late_fuse(datasets, subject_list,
                                                          [kernel_funcs[i_kernel_rgb], kernel_funcs[i_kernel_depth]],
                                                          [Cs[i_C_svm_rgb], Cs[i_C_svm_depth]])
    late_fuse_result = {'weights_sum': weights_sum, 'weights_max': weights_max}
    late_fuse_result = json.dumps(late_fuse_result)
    with open(KINECT_TRAIN + '_' + KERNEL_TYPE + '_late_fuse.json', 'w') as f:
        print('Late fusion:', late_fuse_result)
        f.write(late_fuse_result)
        print()
        print()
        print()


'''
Program begins
'''
for param in dir(CONFIGURATIONS):
    called_param = getattr(CONFIGURATIONS, str(param))
    if isinstance(called_param, GridSearchParams) and called_param.is_assignable(KERNEL_TYPE):
        if not os.path.exists(CONFIGURATIONS.SAVE_DIR):
            os.makedirs(CONFIGURATIONS.SAVE_DIR)
        os.chdir(CONFIGURATIONS.SAVE_DIR)

        print('[MultiviewHandGestures MKL Grid Search]', called_param.name)
        print()
        start_grid_search(gesture_datasets, **called_param.get_params())
        break
