from chemistry.functions.transformations import BaseFunction
from chemistry.functions import GaussianWrapper
import numpy as np


class Path(BaseFunction):
    def __init__(self, charges, charge=0, multiplicity=1, n_proc=2, mem=4100):
        super(Path, self).__init__(len(charges) * 3)
        self.charges = charges
        self.gaussian = GaussianWrapper(n_proc, mem, charge, multiplicity)

    def __call__(self, x):
        en = 0
        if len(x) % self.n_dims == 0:
            for j in range(len(x) // (self.n_dims*3)):
                y = []
                for i in range(self.n_dims):
                    y.append(x[self.n_dims*j+i])
                en += self.gaussian(self.charges, y)
            return en
        else:
            return 'dimension is not correct'

    def value_grad(self, x):
        grad = []
        en = 0
        if len(x) % self.n_dims == 0:
            xarr = []
            for j in range(0, len(x) // self.n_dims):
                y = []
                for i in range(self.n_dims):
                    y.append(x[self.n_dims*j+i])
                xarr.append(np.array(y))
            for i in range(len(xarr)):
                if i == 0 or i == len(xarr)-1:
                    for k in range(len(self.charges)*3):
                        grad.append(0)
                else:
                    doten, dotgrad = self.gaussian.value_grad(self.charges, xarr[i])
                    en += doten
                    uni = (xarr[i + 1] - xarr[i - 1]) / np.linalg.norm(xarr[i + 1] - xarr[i - 1])
                    parallel = 0
                    for k in range(len(dotgrad)):
                        parallel += uni[k]*dotgrad[k]
                    for k in range(len(dotgrad)):
                        grad.append(dotgrad[k] - parallel*uni[k])
            return en, np.array(grad)
        else:
            return 'dimension is not correct'

    def value_grad_hess(self, x):
        return 'not ready'

