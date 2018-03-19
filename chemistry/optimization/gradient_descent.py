from chemistry.utils import io
import numpy as np


class GradientDescent:
    def __init__(self, delta_strategy, stop_strategy):
        self.delta_strategy = delta_strategy
        self.stop_strategy = stop_strategy

    def __call__(self, func, start, path):
        x = np.copy(start)

        itr = 0
        maxval = 0
        trans_state = x
        num_ts = 0
        while True:
            val, grad = func.value_grad(x)

            # only for AFIR
            AFIRval = func.func2(x)
            val -= AFIRval

            if val > maxval:
                maxval = val
                trans_state = x
                num_ts =itr

            path.write("step=" + itr + '\n' + str(x) + '\n'+"energy="+str(val)+'\n')

            delta = self.delta_strategy(itr=iter, x=x, val=val, grad=grad)
            if self.stop_strategy(itr=iter, x=x, val=val, grad=grad, delta=delta):
                path.write("transitional_state="+str(trans_state)+'\n'+"num_ts="+str(num_ts))
                return 0
            x += delta

            itr += 1
