import numpy as np
np.set_printoptions(precision=7)

from chemistry import *
from chemistry.functions import GaussianWrapper
from chemistry.optimization import GradientDescent, delta_strategies, stop_strategies

charges = [1, 1, 8]
eq = np.array([7.0292515e-16,  -1.0996810e+00,   4.5789805e-01,
               -1.4199800e-15, -2.7205852e-01,   1.7920342e+00,
               7.1705485e-16,  -1.2076585e+00,   1.4486635e+00])

molecule = functions.Molecule([1, 1, 8], 3, 750)
optimizer = GradientDescent(delta_strategies.FollowGradient(), stop_strategies.GradNorm(1e-5))
path = optimizer(molecule, eq)

print(utils.io.to_chemcraft_format(charges, eq))

eq = path[-1]
molecule.grad(eq)

molecule = functions.Molecule([1, 1, 8], 3, 750)
transformed = utils.linalg.get_motionless_basis(molecule, eq)

print(molecule.hess(eq))
print()

print(utils.linalg.calc_singular_values(molecule.hess(eq)))
print(utils.linalg.calc_singular_values(transformed.hess(np.zeros(transformed.n_dims))))