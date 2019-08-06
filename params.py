from kernel import Kernel


class Params:

    def __init__(self,
                 name,
                 assignable_names,
                 kinects,
                 kernel_func_mkl_rgb,
                 kernel_func_mkl_depth,
                 kernel_func_concatenate,
                 kernel_func_svm_rgb=None,
                 kernel_func_svm_depth=None,
                 C_mkl=None,
                 C_rgb=None,
                 C_depth=None,
                 C_concatenate=None,
                 lam_mkl=None,
                 late_fusion_weights_sum=[.5, .5],
                 late_fusion_weights_max=[.5, .5]
                 ):
        self.name = name
        self.assignable_names = [assignable_name.lower() for assignable_name in assignable_names]
        self.kinects = kinects
        self.kernel_mkl_rgb = Kernel(kernel_func_mkl_rgb, name, 'rgb')
        self.kernel_mkl_depth = Kernel(kernel_func_mkl_depth, name, 'depth')
        self.kernel_concatenate = Kernel(kernel_func_concatenate, name, '(rgb-depth)')
        if kernel_func_svm_rgb is None:
            self.kernel_svm_rgb = self.kernel_mkl_rgb
        else:
            self.kernel_svm_rgb = Kernel(kernel_func_svm_rgb, name, 'rgb')
        if kernel_func_svm_depth is None:
            self.kernel_svm_depth = self.kernel_mkl_depth
        else:
            self.kernel_svm_depth = Kernel(kernel_func_svm_depth, name, 'depth')
        self.kernels_mkl = [self.kernel_mkl_rgb, self.kernel_mkl_depth]
        self.C_mkl = C_mkl
        self.C_rgb = C_rgb if C_rgb else C_mkl
        self.C_depth = C_depth if C_depth else C_mkl
        self.C_concatenate = C_concatenate if C_concatenate is not None else C_mkl
        self.lam_mkl = lam_mkl

        self.kernels_mkl = [self.kernel_mkl_rgb, self.kernel_mkl_depth]
        self.kernels_svm = [self.kernel_svm_rgb, self.kernel_svm_depth]
        self.C_svms = [self.C_rgb, self.C_depth]
        self.late_fusion_weights_sum = late_fusion_weights_sum
        self.late_fusion_weights_max = late_fusion_weights_max

    def get_params(self):
        return dict(vars(self))

    def is_assignable(self, keyword):
        return keyword.lower() in self.assignable_names
