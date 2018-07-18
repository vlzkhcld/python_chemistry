import numpy as np

from chemistry.optimization import GradientDescent, delta_strategies, stop_strategies
from chemistry.AFIR.path import Path
from chemistry.AFIR.creat_file import creat_energy_file, creat_path_file
from chemistry.functions import GaussianWrapper, Molecule

def second_optimization(num, g, optsteps, molecule):

    charges = molecule.charges
    n_proc = molecule.gaussian.n_proc
    mem = molecule.gaussian.mem
    charge = molecule.gaussian.charge
    multiplicity = molecule.gaussian.multiplicity

    in_path = open(num+'g'+g+'path.xyz', 'r')
    pathlines = in_path.readlines()
    in_path.close()

    path = []
    amount = int(pathlines[0].split()[0])
    steps = len(pathlines) // (amount+2)
    for i in range(steps):
        dot = np.array([])
        for j in range(amount):
            coord = pathlines[i*(amount+2)+2+j].split()
            dot = np.append(dot, float(coord[1]))
            dot = np.append(dot, float(coord[2]))
            dot = np.append(dot, float(coord[3]))
        path.append(dot)

    fastoptimizer = GradientDescent(delta_strategies.FollowGradient(0.2 ), stop_strategies.GradNorm(7e-3))

    startpath, start_en = fastoptimizer(molecule, path[0], 10)

    for dot in startpath[1:]:
        path.insert(0, dot)

    endpath, end_en = fastoptimizer(molecule, path[-1], 30)

    for dot in endpath[1:]:
        path.append(dot)

    lenth = 0
    lenthes = []

    for j in range(len(path)-1):
        dist = np.linalg.norm(path[j+1] - path[j])
        lenthes.append(dist)
        lenth +=dist

    shortpath = [path[0]]

    dist = 0
    for i in range(len(path)-2):
        dist += lenthes[i]
        if dist > 0.15:
            shortpath.append(path[i+1])
            dist = 0
    shortpath.append(path[-1])

    molecule_path = Path(charges, charge, multiplicity, n_proc, mem)

    pathcoord = []

    for x in shortpath:
        for y in x:
            pathcoord.append(y)

    optimizer = GradientDescent(delta_strategies.FollowGradient(0.2), stop_strategies.GradNorm(7e-3))

    pathofpath, enofpath = optimizer(molecule_path, np.array(pathcoord), optsteps)

    goodpathofpath = []

    for x in pathofpath:
        goodpath = []
        itr = 0
        dot = []
        for y in x:
            if itr < len(charges)*3:
                dot.append(y)
                itr += 1
            else:
                goodpath.append(np.array(dot))
                dot = [y]
                itr = 1
        goodpath.append(np.array(dot))
        goodpathofpath.append(goodpath)

    finalen = open(num + 'g' + g + 'final_paths_energies.txt', 'w')
    creat_energy_file(enofpath, finalen)
    finalen.close()
    for i in range(len(goodpathofpath)):
        finalpath = open(num+'g'+g+'final'+str(i)+'path.xyz', 'w')
        creat_path_file(goodpathofpath[i], charges, finalpath)
        finalpath.close()
