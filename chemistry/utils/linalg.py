from math import *
import numpy as np


def eye(n, pos):
    result = np.zeros(n)
    result[pos] = 1
    return result


def construct_rotation_matrix(u, v, alpha=None):
    if np.array_equal(u, v):
        return np.identity(len(u))

    if np.array_equal(u, -v):
        assert alpha is None
        return -np.identity(len(u))

    if alpha is None:
        alpha = acos(u.dot(v) / np.linalg.norm(u) / np.linalg.norm(v))
        u = u / np.linalg.norm(u)
        v = v - u * v.dot(u)
        v /= np.linalg.norm(v)

    A = (np.outer(v, u) - np.outer(u, v))
    B = (np.outer(u, u) + np.outer(v, v))
    return np.identity(len(u)) + sin(alpha) * A + (cos(alpha) - 1) * B


def get_motionless_basis(func, struct):
    directions = []
    for i in range(3):
        translation = np.zeros_like(struct)
        translation[i::3] = 1.
        directions.append(translation / np.linalg.norm(translation))

    for i in range(3):
        rotation = np.zeros_like(struct)
        # rotation[i::3] = struct[i::3]
        rotation[(i + 1) % 3::3] = -struct[(i + 2) % 3::3]
        rotation[(i + 2) % 3::3] = struct[(i + 1) % 3::3]
        directions.append(rotation / np.linalg.norm(rotation))

    basis = np.zeros((func.n_dims, func.n_dims))
    for i, direction in enumerate(directions):
        basis[:, i] = direction

    for i in range(len(directions), func.n_dims):
        new_dir = np.random.randn(func.n_dims)

        new_dir = new_dir - basis[:, :i].dot(np.linalg.lstsq(basis[:, :i], new_dir)[0])
        basis[:, i] = new_dir / np.linalg.norm(new_dir)

    return basis[:, len(directions):]


def get_motionless(func, struct):
    from chemistry.functions import AffineTransformation
    return AffineTransformation(func, struct, get_motionless_basis(func, struct))


def get_normal_coordinates_basis(func, struct):
    from chemistry.functions import AffineTransformation

    motionless_basis = get_motionless_basis(func, struct)
    motionless = AffineTransformation(func, struct, motionless_basis)

    u, s, vh = np.linalg.svd(motionless.hess(np.zeros(motionless.n_dims)))
    normal_basis = u.dot(np.diag(1 / np.sqrt(s)))

    return motionless_basis.dot(normal_basis)


def get_normal_coordinate(func, struct):
    from chemistry.functions import AffineTransformation
    return AffineTransformation(func, struct, get_normal_coordinates_basis(func, struct))


def calc_singular_values(matr):
    u, s, vh = np.linalg.svd(matr)
    return np.diagonal(u.T.dot(matr).dot(u))


if __name__ == '__main__':
    u = np.random.randn(10)
    v = np.random.randn(10)

    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)

    # u = np.array([1., 0, 0])
    # v = np.array([0., 1., 0])

    M = construct_rotation_matrix(u, v)
    print(u)
    print(v)
    # print(M)
    print(M.dot(u / np.linalg.norm(u)))
