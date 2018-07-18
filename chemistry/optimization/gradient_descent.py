from chemistry.utils import io
import numpy as np


class GradientDescent:
    def __init__(self, delta_strategy, stop_strategy):
        self.delta_strategy = delta_strategy
        self.stop_strategy = stop_strategy

    def __call__(self, func, start, steps=50):
        x = np.copy(start)
        path = [np.copy(x)]
        energy = []
        itr = 0
        while itr < steps:
            val, grad = func.value_grad(x)
            energy.append([val, grad])

            delta = self.delta_strategy(itr=iter, x=x, val=val, grad=grad)
            x += delta
            path.append(np.copy(x))

            if self.stop_strategy(itr=iter, x=x, val=val, grad=grad, delta=delta):
                return path, energy
            itr += 1

        return path, energy
