import abc
from chemistry.utils import linalg
import numpy as np
from math import *


class BaseFunction:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def __call__(self, x):
        return self.value_grad(x)[0]

    def grad(self, x):
        return self.value_grad(x)[1]

    def hess(self, x):
        return self.value_grad_hess(x)[2]

    def value_grad(self, x):
        return self(x), self.grad(x)

    def value_grad_hess(self, x):
        return self(x), self.grad(x), self.hess(x)


class CoordTransformation(BaseFunction):
    def __init__(self, n_dims, func):
        super(CoordTransformation, self).__init__(n_dims)
        self.func = func

    @abc.abstractclassmethod
    def transform(self, x):
        pass

    @abc.abstractclassmethod
    def _first_order(self, x):
        """
        :param x:
        :return: J_ij = (\partial{f_j} / \partial{x_i})
        """
        pass

    @abc.abstractclassmethod
    def _second_order(self, x):
        """
        :param x:
        :return: H_ijk = (\partial{f_k} / (\partial{x_i} \partial{x_j}))
        """
        pass

    def __call__(self, x):
        return self.func(self.transform(x))

    def grad(self, x):
        return self._first_order(x).dot(self.func.grad(self.transform(x)))

    def hess(self, x):
        first_order = self._first_order(x)
        second_order = self._second_order(x)
        value, grad, hess = self.func.value_grad_hess(self.transform(x))
        return first_order.dot(hess).dot(first_order.T) + second_order.dot(grad)

    def value_grad(self, x):
        value, grad = self.func.value_grad(self.transform(x))
        return value, self._first_order(x).dot(grad)

    def value_grad_hess(self, x):
        first_order = self._first_order(x)
        second_order = self._second_order(x)
        value, grad, hess = self.func.value_grad_hess(self.transform(x))

        return value, first_order.dot(grad), first_order.dot(hess).dot(first_order.T) + second_order.dot(grad)


class AffineTransformation(CoordTransformation):
    def __init__(self, func, delta=None, basis=None):
        if delta is None:
            delta = np.zeros(func.n_dims)
        if basis is None:
            basis = np.identity(len(delta))

        super(AffineTransformation, self).__init__(basis.shape[1], func)
        self.delta = delta
        self.basis = basis

    def transform(self, x):
        return self.basis.dot(x) + self.delta

    def _first_order(self, x):
        return self.basis.T

    def _second_order(self, x):
        return np.zeros((self.n_dims, self.n_dims, self.func.n_dims))


class PolarCoords(CoordTransformation):
    @staticmethod
    def pcumprod(arr):
        result = np.zeros(len(arr) + 1, arr.dtype)
        result[0] = 1.
        result[1:] = np.cumprod(arr)
        return result

    def __init__(self, func, r):
        super(PolarCoords, self).__init__(func.n_dims - 1, func)
        self.r = r

    def transform(self, phi):
        x = np.zeros(self.n_dims + 1)
        x[:len(phi)] = np.cos(phi)
        x[-1] = 1.
        x[1:] *= np.cumprod(np.sin(phi))

        return self.r * x

    def _first_order(self, phi):
        left_sin_prod = self.pcumprod(np.sin(phi))

        first_order = np.zeros((self.n_dims, self.n_dims + 1))
        for j in range(self.n_dims + 1):
            right_sin_prod = self.pcumprod(np.sin(phi[:j][::-1]))[::-1]
            factor = cos(phi[j]) if j < self.n_dims else 1.

            first_order[:j, j] = factor * left_sin_prod[:j] * np.cos(phi[:j]) * right_sin_prod[1:]

            if j < self.n_dims:
                first_order[j, j] = -left_sin_prod[j + 1]

        return self.r * first_order

    def _second_order(self, phi):
        second_order = np.zeros((self.n_dims, self.n_dims, self.n_dims + 1))

        for i in range(self.n_dims):
            for j in range(self.n_dims):
                k_min = max(i, j)
                k_max = self.func.n_dims - 1

                left_sin_prod = self.pcumprod(np.sin(phi[k_min + 1:]))

                if i != j:
                    prefix = np.sin(phi[:min(i, j)]).prod()
                    prefix *= cos(phi[min(i, j)])
                    prefix *= np.sin(phi[min(i, j) + 1: max(i, j)]).prod()
                    second_order[i, j, k_min] = -prefix * sin(phi[k_min])
                    prefix *= cos(phi[k_min])
                else:
                    prefix = np.sin(phi[:i]).prod()
                    second_order[i, j, k_min] = -prefix * cos(phi[k_min])
                    prefix *= -sin(phi[k_min])

                second_order[i, j, k_min + 1:k_max] = prefix * left_sin_prod[:-1] * np.cos(phi[k_min + 1:])
                second_order[i, j, k_max] = prefix * left_sin_prod[-1]

        return self.r * second_order

    def _slow_second_order(self, phi):
        second_order = np.zeros((self.n_dims, self.n_dims, self.n_dims + 1))

        for i in range(self.n_dims):
            for j in range(self.n_dims):
                for k in range(self.n_dims + 1):
                    if k < max(i, j):
                        continue

                    value = 1.
                    for l in range(min(self.n_dims, k + 1)):
                        count = sum([l == i, l == j, l == k])
                        if count % 2 == 0:
                            value *= sin(phi[l])
                        else:
                            value *= cos(phi[l])
                        if count > 1:
                            value *= -1

                    second_order[i, j, k] = value

        return second_order


class ModelFunction(BaseFunction):
    def __init__(self, n_dims):
        super(ModelFunction, self).__init__(n_dims)

    def __call__(self, x):
        return np.linalg.norm(x) ** 2

    def grad(self, x):
        return 2 * x

    def hess(self, x):
        return 2 * np.identity(self.n_dims)


class PolarCoordsWithDirection(CoordTransformation):
    def __init__(self, func, r, dir):
        super(PolarCoordsWithDirection, self).__init__(func.n_dims - 1, func)

        self.r = r
        self.dir = dir

        M = linalg.construct_rotation_matrix(linalg.eye(func.n_dims, func.n_dims - 1), dir)
        self.affine = AffineTransformation(func, basis=M)
        self.polar = PolarCoords(self.affine, self.r)
        self.zero_angle = .5 * pi * np.ones(self.n_dims)

    def phi_transform(self, phi):
        return phi + self.zero_angle

    def transform(self, phi):
        return self.affine.transform(self.polar.transform(self.phi_transform(phi)))

    def _first_order(self, phi):
        phi = self.phi_transform(phi)
        polar_transform = self.polar.transform(phi)

        J_polar = self.polar._first_order(phi)
        J_affine = self.affine._first_order(polar_transform)

        return J_polar.dot(J_affine)

    def _second_order(self, phi):
        phi = self.phi_transform(phi)
        polar_transform = self.polar.transform(phi)

        J_polar = self.polar._first_order(phi)
        H_polar = self.polar._second_order(phi)

        J_affine = self.affine._first_order(polar_transform)
        H_affine = self.affine._second_order(polar_transform)

        first = np.tensordot(H_polar, J_affine, (2, 0))
        second = np.tensordot(J_polar, np.tensordot(J_polar, H_affine, (1, 0)), (1, 1))

        return first + second


if __name__ == '__main__':
    def test_function_grad(func, lower_bound, upper_bound, n_iters, delta=1e-4, eps=1e-3):
        for _ in range(n_iters):
            x = np.random.uniform(lower_bound, upper_bound)
            ref_grad = func.grad(x)

            for i in range(func.n_dims):
                e = delta * linalg.eye(func.n_dims, i)
                numerical_grad = .5 * (func(x + e) - func(x - e)) / delta
                assert abs(ref_grad[i] - numerical_grad) < eps, '{} : {} vs {}'.format(i, ref_grad[i], numerical_grad)


    def test_function_hess(func, lower_bound, upper_bound, n_iters, delta=1e-4, eps=1e-3):
        for _ in range(n_iters):
            x = np.random.uniform(lower_bound, upper_bound)
            ref_hess = func.hess(x)
            assert np.linalg.norm(ref_hess - ref_hess.T) < eps

            for i in range(func.n_dims):
                for j in range(i, func.n_dims):
                    e1 = delta * linalg.eye(func.n_dims, i)
                    e2 = delta * linalg.eye(func.n_dims, j)

                    numerical_hess = .25 * (func(x + e1 + e2) - func(x - e1 + e2) - func(x + e1 - e2) + func(
                        x - e1 - e2)) / delta ** 2
                    assert abs(ref_hess[i, j] - numerical_hess) < eps, '{} {} : {} vs {}'.format(i, j, ref_hess[i, j],
                                                                                                 numerical_hess)

    np.random.seed(19)

    # ModelFunction <- AffineTransformation
    func = ModelFunction(3)
    func = AffineTransformation(func, np.random.randn(func.n_dims), np.random.randn(func.n_dims, func.n_dims))
    test_function_grad(func, -np.ones(func.n_dims), np.ones(func.n_dims), 10)
    test_function_hess(func, -np.ones(func.n_dims), np.ones(func.n_dims), 10)

    # ModelFunction <- AffineTransformation <- PolarCoords
    func = ModelFunction(5)
    func = AffineTransformation(func, np.random.randn(func.n_dims), np.random.randn(func.n_dims, func.n_dims))
    func = PolarCoords(func, sqrt(2))
    test_function_grad(func, -np.ones(func.n_dims), np.ones(func.n_dims), 10)
    test_function_hess(func, -np.ones(func.n_dims), np.ones(func.n_dims), 10)

    # ModelFunction <- AffineTransformation <- PolarCoordsWithDirection
    func = ModelFunction(5)
    func = AffineTransformation(func, np.random.randn(func.n_dims), np.random.randn(func.n_dims, func.n_dims))
    func = PolarCoordsWithDirection(func, .3, np.random.randn(func.n_dims))
    test_function_grad(func, -np.ones(func.n_dims), np.ones(func.n_dims), 10)
    test_function_hess(func, -np.ones(func.n_dims), np.ones(func.n_dims), 10)
