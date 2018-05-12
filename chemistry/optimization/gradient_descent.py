from chemistry.utils import io
import numpy as np


class GradientDescent:
    def __init__(self, delta_strategy, stop_strategy):
        self.delta_strategy = delta_strategy
        self.stop_strategy = stop_strategy

    def __call__(self, func, start, path, energy):
        x = np.copy(start)

        itr = 0
        while True:
            val, grad = func.value_grad(x)

            path.write(str(func.n_dims // 3) + '\n' + 'opt' + str(itr) + '\n')
            for j in range(func.n_dims // 3):
                path.write(str(func.charges[j]) + '    ' + str(x[3 * j]) + '  ' + str(x[3 * j + 1]) + '  ' + str(
                    x[3 * j + 2]) + '\n')

            energy.write("step=" + str(itr) + '\n'+"norm.grad="+str(np.linalg.norm(grad))+'\n'+"energy="+str(val)+'\n'+'\n')

            delta = self.delta_strategy(itr=iter, x=x, val=val, grad=grad)
            if self.stop_strategy(itr=iter, x=x, val=val, grad=grad, delta=delta) or (itr > 50):
                return x
            x += delta

            itr += 1
