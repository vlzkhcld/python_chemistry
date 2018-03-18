import numpy as np
np.set_printoptions(precision=7)

from chemistry import *
from chemistry.functions import GaussianWrapper
from chemistry.optimization import GradientDescent, delta_strategies, stop_strategies
from chemistry.AFIR.AFIR import AFIRfunction


optimizer = GradientDescent(delta_strategies.FollowGradient(), stop_strategies.GradNorm(1e-5))


charges = [8, 8, 8]
eq = np.array([0.0,  0.0,   0.0,
               4.0, 2.0,   0.0,
               4.0, -2.0, 0.0])
center = 2
radii = np.array([0.66, 0.66, 0.66])

i = 0
for gamma in [50000, 60000]:
    func = AFIRfunction(radii, center, gamma)
    path = open('path'+str(i)+'.txt', 'w')
    path.write("gamma="+str(gamma) + '\n')
    done = optimizer(func, eq, path)
    i += 1
    path.close()

"""molecule = functions.Molecule([1, 1, 8], 3, 750)
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
print(utils.linalg.calc_singular_values(transformed.hess(np.zeros(transformed.n_dims))))"""