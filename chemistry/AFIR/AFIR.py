from chemistry.functions.transformations import BaseFunction
import numpy as np

class AFIRfunction(BaseFunction):
    def __init__(self, radii, center, gamma, p, rho):
        self.radii = radii
        self.center = center
        self.alpha = rho * gamma / (2 ** (1 / 6) - (1 + (1 + gamma / 1.0061) ** (1 / 2)) ** (1 / 6)) / 3.8164
        self.p = p

    def __call__(self, x):
        sum1 = 0
        sum2 = 0
        for i in range(self.center):
            for j in range(self.center, np.size(self.radii)):
                sum1 += (self.radii[i]+self.radii[j]) ** self.p / ((x[3 * i]-x[3 * j]) ** 2+(x[3 * i+1]-x[3 * j+1]) ** 2+(x[3 * i+2]-x[3 * j+2]) ** 2) ** ((self.p-1) / 2)
                sum2 += (self.radii[i]+self.radii[j]) / ((x[3 * i]-x[3 * j]) ** 2+(x[3 * i+1]-x[3 * j+1]) ** 2+(x[3 * i+2]-x[3 * j+2]) ** 2) ** (1 / 2) ** self.p
        return self.alpha * sum1 / sum2


