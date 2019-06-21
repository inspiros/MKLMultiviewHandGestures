from program_config import ProgramConfig
from kernel import Kernel

class Params:

	def __init__(self,
				name,
				assignable_names,
				kernel_func_rgb,
				kernel_func_depth,
				kernel_func_concatenate,
				C_mkl,
				C_rgb=None,
				C_depth=None,
				C_concatenate=None,
				lam_mkl=None):
		self.name = name
		self.assignable_names = [assignable_name.lower() for assignable_name in assignable_names]
		self.kernel_rgb = Kernel(kernel_func_rgb, name, 'rgb')
		self.kernel_depth = Kernel(kernel_func_depth, name, 'depth')
		self.kernel_concatenate = Kernel(kernel_func_concatenate, name, '(rgb-depth)')
		self.C_mkl = C_mkl
		self.C_rgb = C_rgb if C_rgb is not None else C_mkl
		self.C_depth = C_depth if C_depth is not None else C_mkl
		self.C_concatenate = C_concatenate if C_concatenate is not None else C_mkl
		self.lam_mkl = lam_mkl

	def to_program_config(self):
		return ProgramConfig(name=self.name,
			assignable_names=self.assignable_names,
			kernels=[self.kernel_rgb, self.kernel_depth],
			kernel_concatenate=self.kernel_concatenate,
			C_mkl=self.C_mkl,
			C_svms=[self.C_rgb, self.C_depth],
			C_concatenate=self.C_concatenate,
			lam_mkl=self.lam_mkl
			)

