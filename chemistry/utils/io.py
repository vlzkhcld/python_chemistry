def to_chemcraft_format(charges, x, comment=''):
    lines = [str(len(charges)), comment]
    for i in range(len(charges)):
        lines.append('{}\t{:.11f}\t{:.11f}\t{:.11f}'.format(charges[i], *x[i * 3: i * 3 + 3]))
    return '\n'.join(lines)


if __name__ == '__main__':
    import numpy as np

    X = np.array([0.000000000, -0.859799324, 0.835503236,
                  0.000000000, -0.100462324, 1.431546236,
                  0.000000000, -1.619136324, 1.431546236])

    print(to_chemcraft_format([1, 1, 8], X, 'chemcraft format test'))
