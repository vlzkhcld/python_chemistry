import numpy as np
import itertools
from chemistry.functions import PolarCoordsWithDirection


def optimize_on_sphere(func, r, dir, delta_strategy, stop_strategy):
    """
    Optimizes function on sphere surface with center in zero

    :param func: function to optimize
    :param r: radius of sphere to optimize on
    :param dir: initial direction. Vector of norm r
    :param delta_strategy: iteration delta strategy
    :param stop_strategy: iteration stop strategy
    :return: optimization path of directions
    """

    path = []
    phi = np.zeros(func.n_dims - 1)

    for itr in itertools.count():
        path.append(dir)

        in_polar = PolarCoordsWithDirection(func, r, dir)
        value, grad = in_polar.value_grad(phi)

        delta = delta_strategy(itr=itr, x=phi, val=value, grad=grad)

        print(value, dir, np.linalg.norm(grad), np.linalg.norm(delta))

        if stop_strategy(itr=iter, x=phi, val=value, grad=grad, delta=delta):
            break

        dir = in_polar.transform(delta)

    return path

