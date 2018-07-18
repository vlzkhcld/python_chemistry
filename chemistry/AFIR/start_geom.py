import numpy as np

allvdwradii = [0.001, 120, 140, 182, 153, 192, 170, 155, 152, 147, 154, 227, 173, 184, 210, 180, 180, 175, 188, 275,
               231, 211, 206, 201, 196, 191, 186, 182, 163, 140, 139, 187, 211, 185, 190, 185, 202, 303, 249, 242, 236,
               228, 220, 213, 208, 203, 163, 172, 158, 193, 217, 206, 206, 198, 216, 343, 268, 225, 188, 277, 236, 235,
               268, 261, 263, 255, 258, 256, 256, 252, 252, 247, 238, 230, 223, 218, 215, 210, 175, 166, 155, 196, 202,
               220, 348, 283, 235, 210, 200, 186]


def chek_geometry(start, center):
    for i in range(center):
        for j in range(center, len(start) // 3):
            if (start[3*i]-start[3*j])**2 + (start[3*i+1]-start[3*j+1])**2 + (start[3*i+2]-start[3*j+2])**2 < 4:
                return False
    return True


def start_geometries(reagent1, reagent2, base_atoms1, base_atoms2, charges, center, rotation):
    starters = []
    turned1_r2 = np.zeros(len(reagent2))
    for i in range(len(turned1_r2)):
        if i % 3 == 0:
            turned1_r2[i] = reagent2[i+2]
        if i % 3 == 2:
            turned1_r2[i] = -reagent2[i-2]
        if i % 3 == 1:
            turned1_r2[i] = reagent2[i]
    turned2_r2 = np.zeros(len(reagent2))
    for i in range(len(turned2_r2)):
        if i % 3 == 1:
            turned2_r2[i] = reagent2[i+1]
        if i % 3 == 2:
            turned2_r2[i] = -reagent2[i-1]
        if i % 3 == 0:
            turned2_r2[i] = reagent2[i]
    turned3_r2 = np.zeros(len(reagent2))
    for i in range(len(turned3_r2)):
        if i % 3 == 0:
            turned3_r2[i] = reagent2[i + 1]
        if i % 3 == 1:
            turned3_r2[i] = -reagent2[i - 1]
        if i % 3 == 2:
            turned3_r2[i] = reagent2[i]

    for x in [reagent2, turned1_r2, turned2_r2, turned3_r2][0:rotation]:
        for i in base_atoms1:
            for j in base_atoms2:
                lendist = (allvdwradii[charges[j+center]]/100+allvdwradii[charges[i]]/100)*0.8
                for p in range(6):
                    new_reagent2 = np.zeros(len(x))
                    for k in range(len(x)):
                        dist = 0
                        if k % 3 == p % 3 and p < 3:
                            dist += lendist
                        if k % 3 == p % 3 and p >= 3:
                            dist -= lendist
                        new_reagent2[k] = x[k] - x[3 * j + (k % 3)] + reagent1[3 * i + (k % 3)] + dist
                    if chek_geometry(np.append(reagent1, new_reagent2), len(reagent1) // 3):
                            starters.append(np.append(reagent1, new_reagent2))
    return starters



