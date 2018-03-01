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
    skips1 = []
    skips2 = []

    phi = np.zeros(func.n_dims - 1)

    from chemistry.optimization.delta_strategies import Newton, RFO
    newton = Newton()
    rfo = RFO()

    for itr in itertools.count():
        path.append(dir)

        in_polar = PolarCoordsWithDirection(func, r, dir)
        # value, grad = in_polar.value_grad(phi)
        value, grad, hess = in_polar.value_grad_hess(phi)

        skips1.append(in_polar.transform(newton(grad, hess)))
        skips2.append(in_polar.transform(rfo(grad, hess)))

        delta = delta_strategy(itr=itr, x=phi, val=value, grad=grad)

        print(value, dir, np.linalg.norm(grad), np.linalg.norm(delta))

        if stop_strategy(itr=iter, x=phi, val=value, grad=grad, delta=delta):
            break

        dir = in_polar.transform(delta)

    return path, skips1, skips2


def optimize_on_sphere2(func, r, dir, stop_strategy, critical_delta=1e-6):
    from chemistry.optimization.delta_strategies import RFO
    delta_strategy = RFO()

    path = []

    phi = np.zeros(func.n_dims - 1)

    for itr in itertools.count():
        path.append(dir)

        in_polar = PolarCoordsWithDirection(func, r, dir)
        value, grad, hess = in_polar.value_grad_hess(phi)

        delta = delta_strategy(itr=itr, x=phi, val=value, grad=grad, hess=hess)
        print('\n\nnew iteration\nvalue = {}, grad norm = {}, delta norm = {}'.format(value, dir, np.linalg.norm(grad), np.linalg.norm(delta)))

        if stop_strategy(itr=iter, x=phi, val=value, grad=grad, delta=delta):
            break

        while True:
            next_value = in_polar(delta)

            expected = grad.dot(delta) + .5 * delta.dot(hess.dot(delta))
            real = next_value - value

            print('delta norm = {}'.format(np.linalg.norm(delta)))
            print('expected = {}, real = {}, d = {}'.format(expected, real, abs(expected - real) / abs(expected)))
            print()
            if abs(expected - real) / abs(expected) < .3:
                break
            delta *= .5

        dir = in_polar.transform(delta)

    return path

