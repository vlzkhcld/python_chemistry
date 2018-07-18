allradii = [0.001, 31, 28, 128, 96, 84, 73, 71, 66, 57, 58, 166, 141, 121, 111, 107, 105, 102, 106, 203, 176, 170, 160,
            153, 139, 139, 132, 126, 124, 132, 122, 122, 120, 119, 120, 120, 116, 220, 195, 190, 175, 164, 154, 147,
            146, 142, 139, 145, 144, 142, 139, 139, 138, 139, 140, 244, 215, 207, 204, 203, 201, 199, 198, 196, 194,
            192, 192, 189, 190, 187, 187, 175, 170, 162, 151, 144, 141, 136, 136, 132, 145, 146, 148, 140, 150, 150,
            260, 221, 215, 206, 200, 196, 190, 187, 180, 169]


def afir_input(file1, file2, inputgammas, inputbaseatoms1, inputbaseatoms2):
    lines1 = file1.readlines()
    lines2 = file2.readlines()
    reagent1 = []
    reagent2 = []
    charges = []
    for i in range(2, len(lines1)):
        spl = lines1[i].split()
        charges.append(int(spl[0]))
        reagent1.append(float(spl[1]))
        reagent1.append(float(spl[2]))
        reagent1.append(float(spl[3]))
    for i in range(2, len(lines2)):
        spl = lines2[i].split()
        charges.append(int(spl[0]))
        reagent2.append(float(spl[1]))
        reagent2.append(float(spl[2]))
        reagent2.append(float(spl[3]))
    center = len(reagent1) // 3
    radii = []
    for i in charges:
        radii.append(allradii[i]/100)
    gammas = []
    for x in inputgammas.split():
        gammas.append(int(x))
    base_atoms1 = []
    for x in inputbaseatoms1.split():
        base_atoms1.append(int(x))
    base_atoms2 = []
    for x in inputbaseatoms2.split():
        base_atoms2.append(int(x))
    return reagent1, reagent2, charges, radii, center, base_atoms1, base_atoms2, gammas