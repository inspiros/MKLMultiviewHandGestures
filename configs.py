from program_config import ProgramConfig
from params import Params
from kernel import Kernel
from sklearn.metrics import pairwise

DATASET_ROOT = 'dataset'
CONFIGS = []

'''
Linear config
'''
def linear(X, L=None):
	return pairwise.linear_kernel(X, L)

linear_params = Params(name = 'linear',
					assignable_names = ['lin', 'linear'],
					kernel_func_rgb = linear,
					kernel_func_depth = linear,
					kernel_func_concatenate = linear,
					C_mkl = 0.0001,
					C_concatenate = 100
					)

CONFIGS.append(linear_params.to_program_config())


'''
RBF config
'''
def rbf_rgb(X, L=None): #0.000001
	return pairwise.rbf_kernel(X, L, gamma=0.000001)
def rbf_depth(X, L=None): #0.000001
	return pairwise.rbf_kernel(X, L, gamma=0.000001)
def rbf_concatenate(X, L=None):
	return pairwise.rbf_kernel(X, L, gamma=0.1)

rbf_params = Params(name = 'rbf',
					assignable_names = ['rbf', 'gaussian'],
					kernel_func_rgb = rbf_rgb,
					kernel_func_depth = rbf_depth,
					kernel_func_concatenate = rbf_concatenate,
					C_mkl = 100,
					C_concatenate = 100
					)

CONFIGS.append(rbf_params.to_program_config())


'''
Laplacian config
'''
def laplacian_rgb(X, L=None): #0.00001
	return pairwise.laplacian_kernel(X, L, gamma=0.00001)
def laplacian_depth(X, L=None): #0.00001
	return pairwise.laplacian_kernel(X, L, gamma=0.00001)
def laplacian_concatenate(X, L=None):
	return pairwise.laplacian_kernel(X, L, gamma=0.001)

laplacian_params = Params(name = 'laplacian',
					assignable_names = ['lap', 'laplacian'],
					kernel_func_rgb = laplacian_rgb,
					kernel_func_depth = laplacian_depth,
					kernel_func_concatenate = laplacian_concatenate,
					C_mkl = 100,
					C_concatenate = 100
					)

CONFIGS.append(laplacian_params.to_program_config())
