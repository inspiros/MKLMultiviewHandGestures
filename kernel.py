class Kernel:
    """
    Class for encapsulating kernel function.
    """

    def __init__(self, func, func_name=None, name=None):
        self.kernel = func
        self.func_name = func_name
        self.name = name

    def tostring(self):
        return '{' + self.name + ':' + self.func_name + '}'

    @staticmethod
    def stringigfy(kernels):
        string = '{'
        for i in range(len(kernels)):
            string += kernels[i].name + ':' + kernels[i].func_name
            if i < len(kernels) - 1:
                string += ', '
            else:
                string += '}'
        return string

    def __call__(self, X, L=None):
        return self.kernel(X, L)
