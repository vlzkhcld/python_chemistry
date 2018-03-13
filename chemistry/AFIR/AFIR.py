from chemistry.functions.transformations import BaseFunction
import numpy as np


class AFIRfunction(BaseFunction):
    def __init__(self, radii, center, gamma, p, rho):
        super(AFIRfunction, self).__init__(len(radii) * 3)
        self.radii = radii
        self.center = center
        self.alpha = rho * gamma / (2 ** (1 / 6) - (1 + (1 + gamma / 1.0061) ** (1 / 2)) ** (1 / 6)) / 3.8164
        self.p = p

    def __call__(self, x):
        sum1 = 0
        sum2 = 0
        for i in range(self.center):
            for j in range(self.center, self.n_dims // 3):
                sum1 += (self.radii[i] + self.radii[j]) ** self.p / (
                        (x[3 * i] - x[3 * j]) ** 2 + (x[3 * i + 1] - x[3 * j + 1]) ** 2 + (
                        x[3 * i + 2] - x[3 * j + 2]) ** 2) ** ((self.p - 1) / 2)
                sum2 += (self.radii[i] + self.radii[j]) / (
                        (x[3 * i] - x[3 * j]) ** 2 + (x[3 * i + 1] - x[3 * j + 1]) ** 2 + (
                        x[3 * i + 2] - x[3 * j + 2]) ** 2) ** (1 / 2) ** self.p
        return self.alpha * sum1 / sum2

    def value_grad(self, x):
        sum1 = 0
        sum2 = 0
        for i in range(self.center):
            for j in range(self.center, self.n_dims // 3):
                sum1 += (self.radii[i] + self.radii[j]) ** self.p / (
                        (x[3 * i] - x[3 * j]) ** 2 + (x[3 * i + 1] - x[3 * j + 1]) ** 2 + (
                        x[3 * i + 2] - x[3 * j + 2]) ** 2) ** ((self.p - 1) / 2)
                sum2 += (self.radii[i] + self.radii[j]) / (
                        (x[3 * i] - x[3 * j]) ** 2 + (x[3 * i + 1] - x[3 * j + 1]) ** 2 + (
                        x[3 * i + 2] - x[3 * j + 2]) ** 2) ** (1 / 2) ** self.p
        value = self.alpha * sum1 / sum2
        grad = np.zeros(self.n_dims)
        for k in range(self.n_dims // 3):
            sum1_1 = 0
            sum1_2 = 0
            sum1_3 = 0
            sum2_1 = 0
            sum2_2 = 0
            sum2_3 = 0
            if k < self.center:
                for j in range(self.center, self.n_dims // 3):
                    sum1_1 += (1 - self.p) * (x[3 * k] - x[3 * j]) * (self.radii[k] + self.radii[j]) ** self.p / (
                                (x[3 * k] - x[3 * j]) ** 2 + (x[3 * k + 1] - x[3 * j + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * j + 2]) ** 2) ** ((self.p + 1) / 2)
                    sum1_2 += (1 - self.p) * (x[3 * k + 1] - x[3 * j + 1]) * (
                                self.radii[k] + self.radii[j]) ** self.p / (
                                          (x[3 * k] - x[3 * j]) ** 2 + (x[3 * k + 1] - x[3 * j + 1]) ** 2 + (
                                              x[3 * k + 2] - x[3 * j + 2]) ** 2) ** ((self.p + 1) / 2)
                    sum1_3 += (1 - self.p) * (x[3 * k + 2] - x[3 * j + 2]) * (
                                self.radii[k] + self.radii[j]) ** self.p / (
                                          (x[3 * k] - x[3 * j]) ** 2 + (x[3 * k + 1] - x[3 * j + 1]) ** 2 + (
                                              x[3 * k + 2] - x[3 * j + 2]) ** 2) ** ((self.p + 1) / 2)
                    sum2_1 += -self.p * (x[3 * k] - x[3 * j]) * (self.radii[k] + self.radii[j]) ** self.p / (
                                (x[3 * k] - x[3 * j]) ** 2 + (x[3 * k + 1] - x[3 * j + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * j + 2]) ** 2) ** ((self.p + 2) / 2)
                    sum2_2 += -self.p * (x[3 * k + 1] - x[3 * j + 1]) * (self.radii[k] + self.radii[j]) ** self.p / (
                                (x[3 * k] - x[3 * j]) ** 2 + (x[3 * k + 1] - x[3 * j + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * j + 2]) ** 2) ** ((self.p + 2) / 2)
                    sum2_3 += -self.p * (x[3 * k + 2] - x[3 * j + 2]) * (self.radii[k] + self.radii[j]) ** self.p / (
                                (x[3 * k] - x[3 * j]) ** 2 + (x[3 * k + 1] - x[3 * j + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * j + 2]) ** 2) ** ((self.p + 2) / 2)
            else:
                for i in range(self.center):
                    sum1_1 += (1 - self.p) * (x[3 * k] - x[3 * i]) * (self.radii[k] + self.radii[i]) ** self.p / (
                                (x[3 * k] - x[3 * i]) ** 2 + (x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * i + 2]) ** 2) ** ((self.p + 1) / 2)
                    sum1_2 += (1 - self.p) * (x[3 * k + 1] - x[3 * i + 1]) * (
                                self.radii[k] + self.radii[i]) ** self.p / (
                                          (x[3 * k] - x[3 * i]) ** 2 + (x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                              x[3 * k + 2] - x[3 * i + 2]) ** 2) ** ((self.p + 1) / 2)
                    sum1_3 += (1 - self.p) * (x[3 * k + 2] - x[3 * i + 2]) * (
                                self.radii[k] + self.radii[i]) ** self.p / (
                                          (x[3 * k] - x[3 * i]) ** 2 + (x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                              x[3 * k + 2] - x[3 * i + 2]) ** 2) ** ((self.p + 1) / 2)
                    sum2_1 += -self.p * (x[3 * k] - x[3 * i]) * (self.radii[k] + self.radii[i]) ** self.p / (
                                (x[3 * k] - x[3 * i]) ** 2 + (x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * i + 2]) ** 2) ** ((self.p + 2) / 2)
                    sum2_2 += -self.p * (x[3 * k + 1] - x[3 * i + 1]) * (self.radii[k] + self.radii[i]) ** self.p / (
                                (x[3 * k] - x[3 * i]) ** 2 + (x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * i + 2]) ** 2) ** ((self.p + 2) / 2)
                    sum2_3 += -self.p * (x[3 * k + 2] - x[3 * i + 2]) * (self.radii[k] + self.radii[i]) ** self.p / (
                                (x[3 * k] - x[3 * i]) ** 2 + (x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * i + 2]) ** 2) ** ((self.p + 2) / 2)
            grad[k] = self.alpha * (sum1_1 * sum2 - sum2_1 * sum1) / sum2 ** 2
            grad[k + 1] = self.alpha * (sum1_2 * sum2 - sum2_2 * sum1) / sum2 ** 2
            grad[k + 2] = self.alpha * (sum1_3 * sum2 - sum2_3 * sum1) / sum2 ** 2
        return value, grad

    def value_grad_hess(self, x):
        sum1 = 0
        sum2 = 0
        for i in range(self.center):
            for j in range(self.center, self.n_dims // 3):
                sum1 += (self.radii[i] + self.radii[j]) ** self.p / (
                        (x[3 * i] - x[3 * j]) ** 2 + (x[3 * i + 1] - x[3 * j + 1]) ** 2 + (
                        x[3 * i + 2] - x[3 * j + 2]) ** 2) ** ((self.p - 1) / 2)
                sum2 += (self.radii[i] + self.radii[j]) / (
                        (x[3 * i] - x[3 * j]) ** 2 + (x[3 * i + 1] - x[3 * j + 1]) ** 2 + (
                        x[3 * i + 2] - x[3 * j + 2]) ** 2) ** (1 / 2) ** self.p
        value = self.alpha * sum1 / sum2
        grad = np.zeros(self.n_dims)
        hess = np.zeros([self.n_dims, self.n_dims])
        for k in range(self.n_dims // 3):
            sum1_1 = 0
            sum1_2 = 0
            sum1_3 = 0
            sum2_1 = 0
            sum2_2 = 0
            sum2_3 = 0
            if k < self.center:
                for j in range(self.center, self.n_dims // 3):
                    sum1_1 += (1 - self.p) * (x[3 * k] - x[3 * j]) * (self.radii[k] + self.radii[j]) ** self.p / (
                                (x[3 * k] - x[3 * j]) ** 2 + (x[3 * k + 1] - x[3 * j + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * j + 2]) ** 2) ** ((self.p + 1) / 2)
                    sum1_2 += (1 - self.p) * (x[3 * k + 1] - x[3 * j + 1]) * (
                                self.radii[k] + self.radii[j]) ** self.p / (
                                          (x[3 * k] - x[3 * j]) ** 2 + (x[3 * k + 1] - x[3 * j + 1]) ** 2 + (
                                              x[3 * k + 2] - x[3 * j + 2]) ** 2) ** ((self.p + 1) / 2)
                    sum1_3 += (1 - self.p) * (x[3 * k + 2] - x[3 * j + 2]) * (
                                self.radii[k] + self.radii[j]) ** self.p / (
                                          (x[3 * k] - x[3 * j]) ** 2 + (x[3 * k + 1] - x[3 * j + 1]) ** 2 + (
                                              x[3 * k + 2] - x[3 * j + 2]) ** 2) ** ((self.p + 1) / 2)
                    sum2_1 += -self.p * (x[3 * k] - x[3 * j]) * (self.radii[k] + self.radii[j]) ** self.p / (
                                (x[3 * k] - x[3 * j]) ** 2 + (x[3 * k + 1] - x[3 * j + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * j + 2]) ** 2) ** ((self.p + 2) / 2)
                    sum2_2 += -self.p * (x[3 * k + 1] - x[3 * j + 1]) * (self.radii[k] + self.radii[j]) ** self.p / (
                                (x[3 * k] - x[3 * j]) ** 2 + (x[3 * k + 1] - x[3 * j + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * j + 2]) ** 2) ** ((self.p + 2) / 2)
                    sum2_3 += -self.p * (x[3 * k + 2] - x[3 * j + 2]) * (self.radii[k] + self.radii[j]) ** self.p / (
                                (x[3 * k] - x[3 * j]) ** 2 + (x[3 * k + 1] - x[3 * j + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * j + 2]) ** 2) ** ((self.p + 2) / 2)
            else:
                for i in range(self.center):
                    sum1_1 += (1 - self.p) * (x[3 * k] - x[3 * i]) * (self.radii[k] + self.radii[i]) ** self.p / (
                                (x[3 * k] - x[3 * i]) ** 2 + (x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * i + 2]) ** 2) ** ((self.p + 1) / 2)
                    sum1_2 += (1 - self.p) * (x[3 * k + 1] - x[3 * i + 1]) * (
                                self.radii[k] + self.radii[i]) ** self.p / (
                                          (x[3 * k] - x[3 * i]) ** 2 + (x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                              x[3 * k + 2] - x[3 * i + 2]) ** 2) ** ((self.p + 1) / 2)
                    sum1_3 += (1 - self.p) * (x[3 * k + 2] - x[3 * i + 2]) * (
                                self.radii[k] + self.radii[i]) ** self.p / (
                                          (x[3 * k] - x[3 * i]) ** 2 + (x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                              x[3 * k + 2] - x[3 * i + 2]) ** 2) ** ((self.p + 1) / 2)
                    sum2_1 += -self.p * (x[3 * k] - x[3 * i]) * (self.radii[k] + self.radii[i]) ** self.p / (
                                (x[3 * k] - x[3 * i]) ** 2 + (x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * i + 2]) ** 2) ** ((self.p + 2) / 2)
                    sum2_2 += -self.p * (x[3 * k + 1] - x[3 * i + 1]) * (self.radii[k] + self.radii[i]) ** self.p / (
                                (x[3 * k] - x[3 * i]) ** 2 + (x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * i + 2]) ** 2) ** ((self.p + 2) / 2)
                    sum2_3 += -self.p * (x[3 * k + 2] - x[3 * i + 2]) * (self.radii[k] + self.radii[i]) ** self.p / (
                                (x[3 * k] - x[3 * i]) ** 2 + (x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * i + 2]) ** 2) ** ((self.p + 2) / 2)
            grad[k] = self.alpha * (sum1_1 * sum2 - sum2_1 * sum1) / sum2 ** 2
            grad[k + 1] = self.alpha * (sum1_2 * sum2 - sum2_2 * sum1) / sum2 ** 2
            grad[k + 2] = self.alpha * (sum1_3 * sum2 - sum2_3 * sum1) / sum2 ** 2

        return value, grad, hess

