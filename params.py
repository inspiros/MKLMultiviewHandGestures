class Params:

    def __init__(self,
                 name,
                 assignable_names,
                 kinects,
                 kernel_func_mkl_rgb,
                 kernel_func_mkl_of,
                 kernel_func_mkl_depth,
                 kernel_func_concatenate,
                 kernel_func_svm_rgb=None,
                 kernel_func_svm_of=None,
                 kernel_func_svm_depth=None,
                 C_mkl=None,
                 C_rgb=None,
                 C_of=None,
                 C_depth=None,
                 C_concatenate=None,
                 lam_mkl=None,
                 late_fusion_weights_sum=[1., 1., 1.],
                 late_fusion_weights_max=[1., 1., 1.]
                 ):
        self.modalities = ['rgb', 'of', 'depth']
        self.name = name
        self.assignable_names = [assignable_name.lower() for assignable_name in assignable_names]
        self.kinects = kinects
        self.kernel_mkl_rgb = kernel_func_mkl_rgb
        self.kernel_mkl_of = kernel_func_mkl_of
        self.kernel_mkl_depth = kernel_func_mkl_depth
        self.kernel_concatenate = kernel_func_concatenate
        if kernel_func_svm_rgb is None:
            self.kernel_svm_rgb = self.kernel_mkl_rgb
        else:
            self.kernel_svm_rgb = kernel_func_svm_rgb
        if kernel_func_svm_of is None:
            self.kernel_svm_of = self.kernel_mkl_of
        else:
            self.kernel_svm_of = kernel_func_svm_of
        if kernel_func_svm_depth is None:
            self.kernel_svm_depth = self.kernel_mkl_depth
        else:
            self.kernel_svm_depth = kernel_func_svm_depth
        self.C_mkl = C_mkl
        self.C_rgb = C_rgb if C_rgb else C_mkl
        self.C_of = C_of if C_of else C_mkl
        self.C_depth = C_depth if C_depth else C_mkl
        self.C_concatenate = C_concatenate if C_concatenate is not None else C_mkl
        self.lam_mkl = lam_mkl

        self.kernels_mkl = [self.kernel_mkl_rgb, self.kernel_mkl_of, self.kernel_mkl_depth]
        self.kernels_svm = [self.kernel_svm_rgb, self.kernel_svm_of, self.kernel_svm_depth]
        self.C_svms = [self.C_rgb, self.C_of, self.C_depth]
        self.late_fusion_weights_sum = late_fusion_weights_sum
        self.late_fusion_weights_max = late_fusion_weights_max

    def get_params(self):
        return dict(vars(self))

    def is_assignable(self, keyword):
        return keyword.lower() in self.assignable_names

    def filter_modalities(self, modalities):
        if modalities == 'RGB-OF':
            self.modalities.pop(2)
            self.kernels_mkl.pop(2)
            self.kernels_svm.pop(2)
            self.C_svms.pop(2)
            self.late_fusion_weights_sum.pop(2)
            self.late_fusion_weights_max.pop(2)
        elif modalities == 'OF-D':
            self.modalities.pop(0)
            self.kernels_mkl.pop(0)
            self.kernels_svm.pop(0)
            self.C_svms.pop(0)
            self.late_fusion_weights_sum.pop(0)
            self.late_fusion_weights_max.pop(0)
        elif modalities == 'RGB-D':
            self.modalities.pop(1)
            self.kernels_mkl.pop(1)
            self.kernels_svm.pop(1)
            self.C_svms.pop(1)
            self.late_fusion_weights_sum.pop(1)
            self.late_fusion_weights_max.pop(1)
