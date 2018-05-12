import numpy as np


def chek_geometry(start, center):
    for i in range(center):
        for j in range(center, len(start) // 3):
            if (start[3*i]-start[3*j])**2 + (start[3*i+1]-start[3*j+1])**2 + (start[3*i+2]-start[3*j+2])**2 < 4:
                return False
    return True


# base_atom = [number, Van Der Waals radius]
def start_geometries(reagent1, reagent2, base_atoms1, base_atoms2):
    starters = []
    turned_r2 = np.zeros(len(reagent2))
    for i in range(len(turned_r2)):
        if i % 3 == 0:
            turned_r2[i] = reagent2[i+2]
        if i % 3 == 2:
            turned_r2[i] = -reagent2[i-2]
        if i % 3 == 1:
            turned_r2[i] = reagent2[i]
    tturned_r2 = np.zeros(len(reagent2))
    for i in range(len(tturned_r2)):
        if i % 3 == 1:
            tturned_r2[i] = reagent2[i+1]
        if i % 3 == 2:
            tturned_r2[i] = -reagent2[i-1]
        if i % 3 == 0:
            tturned_r2[i] = reagent2[i]
    for x in [turned_r2, tturned_r2, reagent2]:
        for i in base_atoms1:
            for j in base_atoms2:
                lendist = (j[1]+i[1])*0.8
                for p in range(6):
                    new_reagent2 = np.zeros(len(x))
                    for k in range(len(x)):
                        dist = 0
                        if k % 3 == p % 3 and p < 3:
                            dist += lendist
                        if k % 3 == p % 3 and p >= 3:
                            dist -= lendist
                        new_reagent2[k] = x[k] - x[3 * j[0] + (k % 3)] + reagent1[3 * i[0] + (k % 3)] + dist
                    if chek_geometry(np.append(reagent1, new_reagent2), len(reagent1) // 3):
                            starters.append(np.append(reagent1, new_reagent2))
    return starters



