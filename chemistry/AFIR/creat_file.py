import numpy as np


def creat_path_file(path, charges, pathfile, stile='opt'):
    itr = 0
    for x in path:
        pathfile.write(str(len(charges)) + '\n' + stile + str(itr) + '\n')
        for j in range(len(charges)):
            pathfile.write(str(charges[j]) + '    ' + str(x[3 * j]) + '  ' + str(x[3 * j + 1]) + '  ' + str(
                x[3 * j + 2]) + '\n')
        itr += 1


def creat_energy_file(en_grad, energyfile):
    itr = 0
    for x in en_grad:
        energyfile.write("opt_step=" + str(itr) + '\n' + "norm.grad=" + str(np.linalg.norm(x[1])) + '\n' + "energy=" +
                         str(x[0]) + '\n' + '\n')
        itr += 1

def creat_afir_energy_file(en_grad, energyfile):
    itr = 0
    for x in en_grad:
        energyfile.write("afir_step=" + str(itr) + '\n' + "norm.grad=" + str(np.linalg.norm(x[1])) + '\n' +
                         'norm.AFIR.grad=' + str(np.linalg.norm(x[3])) + '\n' + "energy=" + str(x[0]-x[2]) + '\n' +
                         "AFIRenergy=" + str(x[2]) + '\n'+'\n')
        itr += 1