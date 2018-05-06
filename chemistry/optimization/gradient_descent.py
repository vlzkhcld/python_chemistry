from chemistry.utils import io
import numpy as np


class GradientDescent:
    def __init__(self, delta_strategy, stop_strategy):
        self.delta_strategy = delta_strategy
        self.stop_strategy = stop_strategy

    def __call__(self, func, start, path, energy):
        x = np.copy(start)

        stoper = np.zeros(20)
        itr = 0
        while True:
            val, grad = func.value_grad(x)

            stop = 0
            for j in range(np.size(stoper)-1):
                stoper[j] = stoper[j+1]
                stop += stoper[j+1]
            stoper[19] = val
            stop = (stop+val)/20 - val

            path.write(str(func.n_dims // 3) + '\n' + str(itr) + '\n')
            for j in range(func.n_dims // 3):
                path.write(str(func.charges[j]) + '    ' + str(x[3 * j]) + '  ' + str(x[3 * j + 1]) + '  ' + str(
                    x[3 * j + 2]) + '\n')

            energy.write("step=" + str(itr) + '\n'+"norm.grad="+str(np.linalg.norm(grad))+'\n'+"energy="+str(val)+'\n')

            delta = self.delta_strategy(itr=iter, x=x, val=val, grad=grad)
            if self.stop_strategy(itr=iter, x=x, val=val, grad=grad, delta=delta) or (itr > 200):
                return x
            x += delta

            itr += 1
