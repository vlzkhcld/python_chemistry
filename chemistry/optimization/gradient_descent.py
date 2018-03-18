from chemistry.utils import io
import numpy as np


class GradientDescent:
    def __init__(self, delta_strategy, stop_strategy):
        self.delta_strategy = delta_strategy
        self.stop_strategy = stop_strategy

    def __call__(self, func, start, path):
        x = np.copy(start)

        itr = 0
        while True:
            path.write(str(x) + '\n')

            val, grad = func.value_grad(x)
            delta = self.delta_strategy(itr=iter, x=x, val=val, grad=grad)
            if self.stop_strategy(itr=iter, x=x, val=val, grad=grad, delta=delta):
                path.write("steps="+str(itr))
                return 0
            x += delta

            itr += 1
