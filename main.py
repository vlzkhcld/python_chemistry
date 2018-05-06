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

reagent1 = np.array([-3.218811131,  0.287491311,  -4.755094764,
    -4.270658131,  0.318702311,  -4.432272764,
    -2.783629131,  -0.648620689,  -4.398591764,
    -2.685251131,  1.118249311,  -4.268558764,
    -3.060046131,  0.269853311,  -6.205462764,
    -3.451847131,  1.115719311,  -6.553595764])

base_atoms1 = [[0, 1.52], [4, 1.7]]

reagent2 = np.array([-3.474199523,  0.191450334,  -1.76752838])

base_atoms2 = [[0, 1.47]]

charges = [6, 1, 1, 1, 8, 1, 9]

radii = [0.76, 0.31, 0.31, 0.31, 0.66, 0.31, 0.57]

center = 6

molecule = Molecule(charges, 2, 4100)


starters = start_geometries(reagent1, reagent2, base_atoms1, base_atoms2)

gammas = [100]


i = 0
summary = open('summary.txt', 'w')
for start in starters:
    for gamma in gammas:
        func = Sum(molecule, AFIRfunction(radii, center, gamma))
        energy = open(str(i)+'g'+str(gamma)+'energy.txt', 'w')
        path = open(str(i)+'g'+str(gamma)+'path.xyz', 'w')
        energy.write("gamma="+str(gamma) + '\n')
        stop = AFIRoptimizer(func, start, path, energy)
        path.close()
        energy.close()

        optimiz_min = open(str(i)+'g'+str(gamma)+'optimiz_min.xyz', 'w')
        optimiz_en = open(str(i)+'g'+str(gamma)+'optimiz_en.txt', 'w')
        end = optimizer(molecule, stop, optimiz_min, optimiz_en)
        optimiz_min.close()
        optimiz_en.close()
        readoptimiz_en = open(str(i) + 'g' + str(gamma) + 'optimiz_en.txt', 'r')
        lines = readoptimiz_en.readlines()
        last = lines[len(lines)-1]
        summary.write(str(i) + 'g' + str(gamma)+' min ' + last)
        readoptimiz_en.close()
    i += 1
summary.close()
