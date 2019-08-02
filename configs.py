from params import Params
from sklearn.metrics import pairwise

DATASET_ROOT = '/media/inspiros/Shared/datasets/MultiviewGesture'
LAYER = 'fc6'

# Put every configurations here!
'''
Linear config
'''
linear_params = Params(name='linear',
                       assignable_names=['lin', 'linear'],
                       kernel_func_rgb=lambda X, L=None: pairwise.linear_kernel(X, L),
                       kernel_func_depth=lambda X, L=None: pairwise.linear_kernel(X, L),
                       kernel_func_concatenate=lambda X, L=None: pairwise.linear_kernel(X, L),
                       C_mkl=0.0001,
                       C_concatenate=100,
                       lam_mkl=0.0,
                       late_fusion_weight_rgb=0.8,
                       late_fusion_weight_depth=0.2
                       )

'''
RBF config
'''
rbf_params = Params(name='rbf',
                    assignable_names=['rbf', 'gaussian'],
                    kernel_func_rgb=lambda X, L=None: pairwise.rbf_kernel(X, L, gamma=0.000001),
                    kernel_func_depth=lambda X, L=None: pairwise.rbf_kernel(X, L, gamma=0.000001),
                    kernel_func_concatenate=lambda X, L=None: pairwise.rbf_kernel(X, L, gamma=0.1),
                    C_mkl=100,
                    C_concatenate=100,
                    lam_mkl=1.0,
                    late_fusion_weight_rgb=0.8,
                    late_fusion_weight_depth=0.2
                    )

'''
Laplacian config
'''
laplacian_params = Params(name='laplacian',
                          assignable_names=['lap', 'laplacian'],
                          kernel_func_rgb=lambda X, L=None: pairwise.laplacian_kernel(X, L, gamma=0.00001),
                          kernel_func_depth=lambda X, L=None: pairwise.laplacian_kernel(X, L, gamma=0.00001),
                          kernel_func_concatenate=lambda X, L=None: pairwise.laplacian_kernel(X, L, gamma=0.001),
                          C_mkl=100,
                          C_concatenate=100,
                          lam_mkl=0.5,
                          late_fusion_weight_rgb=0.8,
                          late_fusion_weight_depth=0.2
                          )
