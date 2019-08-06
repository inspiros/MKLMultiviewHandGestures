from gridsearch_params import GridSearchParams
from sklearn.metrics import pairwise
from easydict import EasyDict

DATASET_ROOT = '/media/inspiros/Shared/datasets/MultiviewGesture'
LAYER = 'fc6'
SAVE_DIR = 'gridsearch'

HYPERPARAMETERS = EasyDict(d={
    'Cs': [
        0.0000001,
        0.000001,
        0.00001,
        0.0001,
        0.001,
        0.01,
        0.1,
        1.,
        10.,
        100.,
        1000.
    ],
    'lams': [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.],
    'kernel_funcs': {
        'linear': [lambda X, L=None: pairwise.linear_kernel(X, L)],
        'rbf': [
            lambda X, L=None: pairwise.rbf_kernel(X, L, gamma=0.0000001),
            lambda X, L=None: pairwise.rbf_kernel(X, L, gamma=0.000001),
            lambda X, L=None: pairwise.rbf_kernel(X, L, gamma=0.00001),
            lambda X, L=None: pairwise.rbf_kernel(X, L, gamma=0.0001),
            lambda X, L=None: pairwise.rbf_kernel(X, L, gamma=0.001),
            lambda X, L=None: pairwise.rbf_kernel(X, L, gamma=0.01),
            lambda X, L=None: pairwise.rbf_kernel(X, L, gamma=0.1)
        ],
        'laplacian': [
            lambda X, L=None: pairwise.laplacian_kernel(X, L, gamma=0.0000001),
            lambda X, L=None: pairwise.laplacian_kernel(X, L, gamma=0.000001),
            lambda X, L=None: pairwise.laplacian_kernel(X, L, gamma=0.00001),
            lambda X, L=None: pairwise.laplacian_kernel(X, L, gamma=0.0001),
            lambda X, L=None: pairwise.laplacian_kernel(X, L, gamma=0.001),
            lambda X, L=None: pairwise.laplacian_kernel(X, L, gamma=0.01),
            lambda X, L=None: pairwise.laplacian_kernel(X, L, gamma=0.1)
        ]
    }
})

'''
Linear config
'''
linear_params = GridSearchParams(name='linear',
                                 assignable_names=['lin', 'linear'],
                                 kernel_funcs=HYPERPARAMETERS.kernel_funcs.linear,
                                 Cs=HYPERPARAMETERS.Cs,
                                 lams_mkl=HYPERPARAMETERS.lams
                                 )

'''
RBF
'''
rbf_params = GridSearchParams(name='rbf',
                              assignable_names=['rbf', 'gaussian'],
                              kernel_funcs=HYPERPARAMETERS.kernel_funcs.rbf,
                              Cs=HYPERPARAMETERS.Cs,
                              lams_mkl=HYPERPARAMETERS.lams
                              )

'''
Laplacian
'''
laplacian_params = GridSearchParams(name='laplacian',
                                    assignable_names=['lap', 'laplacian'],
                                    kernel_funcs=HYPERPARAMETERS.kernel_funcs.laplacian,
                                    Cs=HYPERPARAMETERS.Cs,
                                    lams_mkl=HYPERPARAMETERS.lams
                                    )
