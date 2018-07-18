import numpy as np
np.set_printoptions(precision=7)

from chemistry import *
from chemistry.functions import GaussianWrapper, Molecule, Sum
from chemistry.optimization import GradientDescent, delta_strategies, stop_strategies
from chemistry.AFIR.AFIR import AFIRfunction
from chemistry.AFIR.afir_gradient_descent import AFIRGradientDescent
from chemistry.AFIR.start_geom import start_geometries, chek_geometry
from chemistry.AFIR.afir_input import afir_input
from chemistry.AFIR.second_optimization import second_optimization
from chemistry.AFIR.creat_file import creat_energy_file, creat_path_file, creat_afir_energy_file


# input
file1 = open('reagent1.xyz', 'r')
file2 = open('reagent2.xyz', 'r')
print('Enter full charge')
charge = int(input())
print('Enter full multiplicity')
multiplicity = int(input())
print('Enter gamma(s)')
inputgammas = input()
print('Enter base atom(s) of reagent 1')
inputbaseatoms1 = input()
print('Enter base atom(s) of reagent 2')
inputbaseatoms2 = input()
reagent1, reagent2, charges, radii, center, base_atoms1, base_atoms2, gammas = afir_input(file1, file2, inputgammas,
                                                                                          inputbaseatoms1,
                                                                                          inputbaseatoms2)
file1.close()
file2.close()

# minimization strategy for molecules + AFIR
AFIRoptimizer = AFIRGradientDescent(delta_strategies.FollowGradient(0.2), stop_strategies.GradNorm(7e-3))

# minimization strategy for molecules
optimizer = GradientDescent(delta_strategies.FollowGradient(0.2), stop_strategies.GradNorm(7e-3))

# gaussian wrapper
molecule = Molecule(charges, charge, multiplicity, 2, 4100)


print('Do you want to rotate reagent2 in start geometries generation?(1/2/3/4)')
rotation = int(input())

# creation start geometries
starters = start_geometries(reagent1, reagent2, base_atoms1, base_atoms2, charges, center, rotation)

# creations AFIR paths
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

# optimization AFIR path
answer = ' '
while answer != 'y' and answer != 'n':
    print('Do you want to star optimization AFIR paths?(y/n)')
    answer = input()

while answer == 'y':
    print('Enter number of path')
    num = input()
    print('Enter gamma of path')
    g = input()
    print('Enter number of steps of second optimization')
    steps = int(input())
    second_optimization(num, g, steps, molecule)
    print('Do you want to continue?(y/n)')
    answer = input()
