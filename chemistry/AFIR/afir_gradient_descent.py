import numpy as np


class AFIRGradientDescent:
    def __init__(self, delta_strategy, stop_strategy):
        self.delta_strategy = delta_strategy
        self.stop_strategy = stop_strategy

    def __call__(self, func, start, steps_after_ts=50):
        x = np.copy(start)
        energy = []
        path = [np.copy(x)]

        itr = 0
        maxval = -100000
        stoper = 0
        while stoper < steps_after_ts:

            val, grad = func.func1.value_grad(x)
            AFIRval, AFIRgrad = func.func2.value_grad(x)

            if val > maxval:
                maxval = val
                stoper = 0

            grad += AFIRgrad
            val += AFIRval
            stoper += 1

            energy.append([val, grad, AFIRval, AFIRgrad])

            delta = self.delta_strategy(itr=iter, x=x, val=val, grad=grad)
            x += delta
            path.append(np.copy(x))

            if self.stop_strategy(itr=iter, x=x, val=val, grad=grad, delta=delta):
                return path, energy
            itr += 1
        return path, energy