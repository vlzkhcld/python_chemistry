from chemistry.utils import io


class GradientDescent:
    def __init__(self, delta_strategy, stop_strategy):
        self.delta_strategy = delta_strategy
        self.stop_strategy = stop_strategy

    def __call__(self, func, x):
        path = []

        itr = 0
        while True:
            path.append(x)

            val, grad = func.value_grad(x)
            delta = self.delta_strategy(itr=iter, x=x, val=val, grad=grad)
            if self.stop_strategy(itr=iter, x=x, val=val, grad=grad, delta=delta):
                return path
            x += delta

            itr += 1
