import numpy as np


class AFIRGradientDescent:
    def __init__(self, delta_strategy, stop_strategy):
        self.delta_strategy = delta_strategy
        self.stop_strategy = stop_strategy

    def __call__(self, func, start, path, energy):
        x = np.copy(start)

        itr = 0
        maxval = -100000
        trans_state = np.copy(start)
        num_ts = 0
        stoper = 0
        while True:

            val, grad = func.func1.value_grad(x)
            AFIRval, AFIRgrad = func.func2.value_grad(x)

            if val > maxval:
                maxval = val
                trans_state = np.copy(x)
                num_ts = itr
                stoper = 0

            grad += AFIRgrad
            val += AFIRval
            stoper += 1

            energy.write("step=" + str(itr) + '\n' + "norm.grad=" + str(np.linalg.norm(grad)) + '\n' +
                         'norm.AFIR.grad=' + str(np.linalg.norm(AFIRgrad)) + '\n' + "energy=" + str(
                val - AFIRval) + '\n' +
                         "AFIRenergy=" + str(AFIRval) + '\n'+'\n')

            path.write(str(func.n_dims // 3) + '\n' + str(itr) + '\n')
            for j in range(func.n_dims // 3):
                path.write(str(func.func1.charges[j]) + '    ' + str(x[3 * j]) + '  ' + str(x[3 * j + 1]) + '  ' + str(
                    x[3 * j + 2]) + '\n')

            delta = self.delta_strategy(itr=iter, x=x, val=val, grad=grad)
            if self.stop_strategy(itr=iter, x=x, val=val, grad=grad, delta=delta) or stoper > 100:
                energy.write("transitional_state=" + str(trans_state) + '\n' + "num_ts=" + str(num_ts))
                return x
            x += delta

            itr += 1
