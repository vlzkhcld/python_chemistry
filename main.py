import numpy as np
np.set_printoptions(precision=7)

from chemistry import *
from chemistry.functions import GaussianWrapper, Molecule, Sum
from chemistry.optimization import GradientDescent, delta_strategies, stop_strategies
from chemistry.AFIR.AFIR import AFIRfunction
from chemistry.AFIR.afir_gradient_descent import AFIRGradientDescent
from chemistry.AFIR.start_geom import start_geometries, chek_geometry


AFIRoptimizer = AFIRGradientDescent(delta_strategies.FollowGradient(0.1), stop_strategies.GradNorm(7e-3))

optimizer = GradientDescent(delta_strategies.FollowGradient(0.1), stop_strategies.GradNorm(7e-3))

'''reagent1 = np.array([-0.664704000,     -0.007633000,      0.000000000,
       -1.238637000,      0.914145000,      0.000000000,
       -1.238637000,     -0.929411000,      0.000000000,
        0.664704000,     -0.007633000,      0.000000000,
        1.238637000,      0.914145000,      0.000000000,
        1.238637000,     -0.929411000,      0.000000000])

base_atoms1 = [[0, 1.7]]

reagent2 = np.array([0.0,     0.0,      0.373,
                    0.0,     0.0,     -0.373])

base_atoms2 = [[1, 1.2]]

charges = [6, 1, 1, 6, 1, 1, 1, 1]

radii = [0.66, 0.31, 0.31, 0.66, 0.31, 0.31, 0.31, 0.31]

center = 6'''

reagent1 = np.array([-0.664704000,     -0.007633000,      0.000000000,
                    -1.238637000,      0.914145000,      0.000000000,
                    -1.238637000,     -0.929411000,      0.000000000,
                    0.664704000,     -0.007633000,      0.000000000,
                    1.238637000,      0.914145000,      0.000000000,
                    1.238637000,     -0.929411000,      0.000000000])

reagent2 = np.array([0.0, 0.0, 0.0,
                    1.08196, 0.0, 0.0,
                    -0.54098, 0.937, 0.0,
                    -0.54098, -0.937, 0.0])

charges = [6, 1, 1, 6, 1, 1, 6, 1, 1, 1]

base_atoms1 = [[0, 1.7]]

base_atoms2 = [[0, 1.7]]

center = 6

radii = [0.66, 0.31, 0.31, 0.66, 0.31, 0.31, 0.66, 0.31, 0.31, 0.31]

molecule = Molecule(charges, 2, 4100)


starters = start_geometries(reagent1, reagent2, base_atoms1, base_atoms2)

gammas = [1250]


i = 0
summary = open('summary.txt', 'w')
for start in starters:
    for gamma in gammas:
        summary.write(str(i) + 'g' + str(gamma) + '\n' + 'start energy' + str(molecule(start)) + '\n')
        func = Sum(molecule, AFIRfunction(radii, center, gamma))
        energy = open(str(i)+'g'+str(gamma)+'energy.txt', 'w')
        path = open(str(i)+'g'+str(gamma)+'path.xyz', 'w')
        energy.write("gamma="+str(gamma) + '\n')
        stop = AFIRoptimizer(func, start, path, energy)
        path.close()
        energy.close()

        optimiz_min = open(str(i)+'g'+str(gamma)+'path.xyz', 'a')
        optimiz_en = open(str(i)+'g'+str(gamma)+'optimiz_en.txt', 'w')
        end = optimizer(molecule, stop, optimiz_min, optimiz_en)
        optimiz_min.close()
        optimiz_en.close()
        readoptimiz_en = open(str(i) + 'g' + str(gamma) + 'optimiz_en.txt', 'r')
        lines = readoptimiz_en.readlines()
        last = lines[len(lines)-2]
        summary.write(' min ' + last + '\n')
        readoptimiz_en.close()
    i += 1
summary.close()
