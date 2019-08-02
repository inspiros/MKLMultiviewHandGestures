from kernel import Kernel


class Params:

    def __init__(self,
                 name,
                 assignable_names,
                 kernel_func_rgb,
                 kernel_func_depth,
                 kernel_func_concatenate,
                 C_mkl=None,
                 C_rgb=None,
                 C_depth=None,
                 C_concatenate=None,
                 lam_mkl=None,
                 late_fusion_weight_rgb=1,
                 late_fusion_weight_depth=1,
                 ):
        self.name = name
        self.assignable_names = [assignable_name.lower() for assignable_name in assignable_names]
        self.kernel_rgb = Kernel(kernel_func_rgb, name, 'rgb')
        self.kernel_depth = Kernel(kernel_func_depth, name, 'depth')
        self.kernel_concatenate = Kernel(kernel_func_concatenate, name, '(rgb-depth)')
        self.kernels = [self.kernel_rgb, self.kernel_depth]
        self.C_mkl = C_mkl
        self.C_rgb = C_rgb if C_rgb else C_mkl
        self.C_depth = C_depth if C_depth else C_mkl
        self.C_concatenate = C_concatenate if C_concatenate is not None else C_mkl
        self.lam_mkl = lam_mkl
        self.late_fusion_weight_rgb = late_fusion_weight_rgb / (late_fusion_weight_rgb + late_fusion_weight_depth)
        self.late_fusion_weight_depth = late_fusion_weight_depth / (late_fusion_weight_rgb + late_fusion_weight_depth)

        self.kernels = [self.kernel_rgb, self.kernel_depth]
        self.C_svms = [self.C_rgb, self.C_depth]
        self.late_fusion_weights = [self.late_fusion_weight_rgb, self.late_fusion_weight_depth]

    def get_params(self):
        return dict(vars(self))

    def is_assignable(self, keyword):
        return keyword.lower() in self.assignable_names
