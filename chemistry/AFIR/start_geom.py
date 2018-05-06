import numpy as np


def chek_geometry(start, center):
    for i in range(center):
        for j in range(center, len(start) // 3):
            if (start[i]-start[j])**2 + (start[i+1]-start[j+1])**2 + (start[i+2]-start[j+2])**2 < 4:
                return False
    return True


def start_geometries(reagent1, reagent2, base_atoms1, base_atoms2):
    starters = []
    for i in base_atoms1:
        for j in base_atoms2:
            lendist = (j[1]+i[1])*1.1
            for p in range(6):
                new_reagent2 = np.zeros(len(reagent2))
                for k in range(len(reagent2)):
                    dist = 0
                    if k % 3 == p % 3 and p < 3:
                        dist += lendist
                    if k % 3 == p % 3 and p >= 3:
                        dist -= lendist
                    new_reagent2[k] = reagent2[k] - reagent2[3 * j[0] + (k % 3)] + reagent1[3 * i[0] + (k % 3)] + dist
                if chek_geometry(np.append(reagent1, new_reagent2), len(reagent1)):
                        starters.append(np.append(reagent1, new_reagent2))
    return starters



