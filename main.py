import numpy as np
np.set_printoptions(precision=7)

from chemistry import *
from chemistry.functions import GaussianWrapper, Molecule, Sum
from chemistry.optimization import GradientDescent, delta_strategies, stop_strategies
from chemistry.AFIR.AFIR import AFIRfunction


optimizer = GradientDescent(delta_strategies.FollowGradient(), stop_strategies.GradNorm(1e-5))


charges = [6, 1, 1, 1, 8, 1, 9]
eq = np.array([-3.245201265,      0.277567077,     -4.446379571,
               -4.297048265,      0.308778077,     -4.123557571,
               -2.810019265,     -0.658544923,     -4.089876571,
               -2.711641265,      1.108325077,     -3.959843571,
               -3.086436265,      0.259929077,     -5.896747571,
               -3.478000000,      1.106000000,    -6.245000000,
               -4.572820000,      3.468910000,    -7.217470000])
center = 6
radii = np.array([0.76, 0.31, 0.31, 0.31, 0.66, 0.31, 0.57])

i = 0
for gamma in [50000]:
    func = Sum(AFIRfunction(radii, center, gamma), functions.Molecule(charges, 4, 2000))
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