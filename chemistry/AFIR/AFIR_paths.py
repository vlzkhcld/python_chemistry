import numpy as np

from chemistry.functions import Sum
from chemistry.optimization import GradientDescent, delta_strategies, stop_strategies
from chemistry.AFIR.AFIR import AFIRfunction
from chemistry.AFIR.afir_gradient_descent import AFIRGradientDescent
from chemistry.AFIR.creat_file import creat_energy_file, creat_path_file, creat_afir_energy_file

# minimization strategy for molecules + AFIR
AFIRoptimizer = AFIRGradientDescent(delta_strategies.FollowGradient(0.2), stop_strategies.GradNorm(7e-3))

# minimization strategy for molecules
optimizer = GradientDescent(delta_strategies.FollowGradient(0.2), stop_strategies.GradNorm(7e-3))


def creat_AFIR_paths(starters, gammas, molecule, radii, center):
    charges = molecule.charges
    i = 0
    summary = open('summary.txt', 'w')
    for start in starters:
        for gamma in gammas:
            summary.write(str(i) + 'g' + str(gamma) + '\n' + 'start energy' + str(molecule(start)) + '\n')
            func = Sum(molecule, AFIRfunction(radii, center, gamma))

            path, energy = AFIRoptimizer(func, start, 30)
            file_energy = open(str(i)+'g'+str(gamma)+'energy.txt', 'w')
            file_path = open(str(i)+'g'+str(gamma)+'path.xyz', 'w')
            file_energy.write("gamma="+str(gamma) + '\n')
            creat_afir_energy_file(energy, file_energy)
            creat_path_file(path, charges, file_path, 'afir')
            file_path.close()
            file_energy.close()

            optimiz_path, optimiz_en = optimizer(molecule, path[-1], 10)
            file_optimiz_path = open(str(i)+'g'+str(gamma)+'path.xyz', 'a')
            file_optimiz_en = open(str(i)+'g'+str(gamma)+'energy.txt', 'a')
            creat_path_file(optimiz_path, charges, file_optimiz_path)
            creat_energy_file(optimiz_en, file_optimiz_en)
            file_optimiz_path.close()
            file_optimiz_en.close()
            readoptimiz_en = open(str(i) + 'g' + str(gamma) + 'energy.txt', 'r')
            lines = readoptimiz_en.readlines()
            last = lines[len(lines)-2]
            summary.write(' min ' + last + '\n')
            readoptimiz_en.close()
        i += 1
    summary.close()
