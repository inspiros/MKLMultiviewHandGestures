
class ProgramConfig:
	'''
	'''
	def __init__(self, name, assignable_names, kernels, kernel_concatenate, C_mkl=None, C_svms=None, C_concatenate=None, lam_mkl=None):
		self.name = name
		self.assignable_names = assignable_names
		self.kernels = kernels
		self.kernel_concatenate = kernel_concatenate
		self.C_mkl = C_mkl
		self.C_svms = C_svms if C_svms is not None else [C_mkl for k in kernels]
		self.C_concatenate = C_concatenate
		self.lam_mkl = lam_mkl

	def is_assignable(self, keyword):
		return keyword.lower() in self.assignable_names

	def to_params(self):
		return (self.kernels, self.kernel_concatenate, self.C_mkl, self.C_svms, self.C_concatenate, self.lam_mkl)
