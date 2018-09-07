from chemistry.functions import Molecule
from chemistry.AFIR.start_geom import start_geometries
from chemistry.AFIR.afir_input import afir_input
from chemistry.AFIR.optimization import optimization
from chemistry.AFIR.creat_file import creat_path_file
from chemistry.AFIR.AFIR_paths import creat_AFIR_paths

import numpy as np
import os
np.set_printoptions(precision=7)


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

# gaussian wrapper
molecule = Molecule(charges, charge, multiplicity, 2, 4100)

# creation start geometries
all_starters = start_geometries(reagent1, reagent2, base_atoms1, base_atoms2, charges, center)
starters_file = open('starters.xyz', 'w')
creat_path_file(all_starters, charges, starters_file, 'start_geometry')
starters_file.close()

print('Choose start geometries. You can check them out in starters.xyz file.')
num_starters = input().split()
starters = []
for i in num_starters:
    starters.append(all_starters[int(i)])


# creations AFIR paths
os.mkdir('AFIR paths')
os.chdir('./AFIR paths')
creat_AFIR_paths(starters, gammas, molecule, radii, center)
os.chdir('..')

# optimization AFIR path
answer = ' '
while answer != 'y' and answer != 'n':
    print('Do you want to start optimization AFIR paths?(y/n)')
    answer = input()
if answer == 'y':
    os.mkdir('optimization of paths')
    os.chdir('./optimization of paths')
while answer == 'y':
    print('Enter number of path')
    num = input()
    print('Enter gamma of path')
    g = input()
    print('Enter number of steps of second optimization')
    steps = int(input())
    os.mkdir(num+'g'+g)
    os.chdir('./'+num+'g'+g)
    optimization(num, g, steps, molecule)
    os.chdir('..')
    answer = ' '
    while answer != 'y' and answer != 'n':
        print('Do you want to continue?(y/n)')
        answer = input()
    if answer == 'n':
        os.chdir('..')


