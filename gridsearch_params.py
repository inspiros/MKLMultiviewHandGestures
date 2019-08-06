from kernel import Kernel
from sklearn.preprocessing import normalize
import numpy as np


class GridSearchParams:

    def __init__(self,
                 name,
                 assignable_names,
                 kernel_funcs,
                 Cs=[],
                 lams_mkl=[],
                 ):
        self.name = name
        self.assignable_names = [assignable_name.lower() for assignable_name in assignable_names]
        self.kernel_funcs = kernel_funcs
        self.kernels = [kernel_funcs, kernel_funcs]
        self.Cs = Cs
        self.lams_mkl = lams_mkl
        self.Cs_svms = [Cs, Cs]

    def get_params(self):
        return dict(vars(self))

    def is_assignable(self, keyword):
        return keyword.lower() in self.assignable_names
