from chemistry.functions.transformations import BaseFunction
import numpy as np


class AFIRfunction(BaseFunction):
    def __init__(self, radii, center, gamma, p=6, rho=1):
        super(AFIRfunction, self).__init__(len(radii) * 3)
        self.radii = radii
        self.center = center
        self.alpha = 3.808799*10**(-7)*rho * gamma / (2 ** (1 / 6) - (1 + (1 + gamma / 1.0061) ** (1 / 2)) ** (1 / 6)) / 3.8164
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

            #print("k=",k,"sum1=",sum1,"sum2=",sum2,"sum1_1=",sum1_1,"sum1_2=",sum1_2,"sum1_3=",sum1_3,"sum2_1=",sum2_1,"sum2_2=",sum2_2,"sum2_3=",sum2_3)
            grad[3*k] = self.alpha * (sum1_1 * sum2 - sum2_1 * sum1) / sum2 ** 2
            grad[3*k + 1] = self.alpha * (sum1_2 * sum2 - sum2_2 * sum1) / sum2 ** 2
            grad[3*k + 2] = self.alpha * (sum1_3 * sum2 - sum2_3 * sum1) / sum2 ** 2
            #print("k="+str(k), "grad=("+str(grad[3*k])+"; "+str(grad[3*k+1])+"; "+str(grad[3*k+2])+")")
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
            for p in range(k + 1):
                sum1_1_1 = 0
                sum1_1_2 = 0
                sum1_1_3 = 0
                sum1_2_2 = 0
                sum1_2_3 = 0
                sum1_3_3 = 0
                sum2_1_1 = 0
                sum2_1_2 = 0
                sum2_1_3 = 0
                sum2_2_2 = 0
                sum2_2_3 = 0
                sum2_3_3 = 0
                sum1_id_id = 0
                sum2_id_id = 0

                if p == k:
                    if p < self.center:
                        for j in range(self.center, self.n_dims // 3):
                            sum1_id_id += (1 - self.p) * ((self.radii[k] + self.radii[j]) ** self.p / (
                                    (x[3 * k] - x[3 * j]) ** 2 + (x[3 * k + 1] - x[3 * j + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * j + 2]) ** 2) ** ((self.p + 1) / 2))
                            sum1_1_1 += -(1 - self.p) * (self.p + 1) * (x[3 * k] - x[3 * j]) ** 2 * (
                                    self.radii[k] + self.radii[i]) ** self.p / ((x[3 * k] - x[3 * i]) ** 2 + (
                                    x[3 * k + 1] - x[3 * i + 1]) ** 2 + (x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 3) / 2)
                            sum1_1_2 += -(1 - self.p) * (self.p + 1) * (x[3 * k] - x[3 * j]) * (
                                    x[3 * k + 1] - x[3 * j + 1]) * (self.radii[k] + self.radii[i]) ** self.p / (
                                                (x[3 * k] - x[3 * i]) ** 2 + (
                                                x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                                        x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 3) / 2)
                            sum1_1_3 += -(1 - self.p) * (self.p + 1) * (x[3 * k] - x[3 * j]) * (
                                    x[3 * k + 2] - x[3 * j + 2]) * (self.radii[k] + self.radii[i]) ** self.p / (
                                                (x[3 * k] - x[3 * i]) ** 2 + (
                                                x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                                        x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 3) / 2)
                            sum1_2_2 += -(1 - self.p) * (self.p + 1) * (x[3 * k + 1] - x[3 * j + 1]) ** 2 * (
                                    self.radii[k] + self.radii[i]) ** self.p / ((x[3 * k] - x[3 * i]) ** 2 + (
                                    x[3 * k + 1] - x[3 * i + 1]) ** 2 + (x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 3) / 2)
                            sum1_2_3 += -(1 - self.p) * (self.p + 1) * (x[3 * k + 1] - x[3 * j + 1]) * (
                                    x[3 * k + 2] - x[3 * j + 2]) * (self.radii[k] + self.radii[i]) ** self.p / (
                                                (x[3 * k] - x[3 * i]) ** 2 + (
                                                x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                                        x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 3) / 2)
                            sum1_3_3 += -(1 - self.p) * (self.p + 1) * (x[3 * k + 2] - x[3 * j + 2]) ** 2 * (
                                    self.radii[k] + self.radii[i]) ** self.p / ((x[3 * k] - x[3 * i]) ** 2 + (
                                    x[3 * k + 1] - x[3 * i + 1]) ** 2 + (x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 3) / 2)
                            sum2_id_id += -self.p * ((self.radii[k] + self.radii[j]) ** self.p / (
                                    (x[3 * k] - x[3 * j]) ** 2 + (x[3 * k + 1] - x[3 * j + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * j + 2]) ** 2) ** ((self.p + 2) / 2))
                            sum2_1_1 += -self.p * (self.p + 2) * (x[3 * k] - x[3 * j]) ** 2 * (
                                    self.radii[k] + self.radii[i]) ** self.p / ((x[3 * k] - x[3 * i]) ** 2 + (
                                    x[3 * k + 1] - x[3 * i + 1]) ** 2 + (x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 4) / 2)
                            sum2_1_2 += -self.p * (self.p + 2) * (x[3 * k] - x[3 * j]) * (
                                    x[3 * k + 1] - x[3 * j + 1]) * (self.radii[k] + self.radii[i]) ** self.p / (
                                                (x[3 * k] - x[3 * i]) ** 2 + (
                                                x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                                        x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 4) / 2)
                            sum2_1_3 += -self.p * (self.p + 2) * (x[3 * k] - x[3 * j]) * (
                                    x[3 * k + 2] - x[3 * j + 2]) * (self.radii[k] + self.radii[i]) ** self.p / (
                                                (x[3 * k] - x[3 * i]) ** 2 + (
                                                x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                                        x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 4) / 2)
                            sum2_2_2 += -self.p * (self.p + 2) * (x[3 * k + 1] - x[3 * j + 1]) ** 2 * (
                                    self.radii[k] + self.radii[i]) ** self.p / ((x[3 * k] - x[3 * i]) ** 2 + (
                                    x[3 * k + 1] - x[3 * i + 1]) ** 2 + (x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 4) / 2)
                            sum2_2_3 += -self.p * (self.p + 2) * (x[3 * k + 1] - x[3 * j + 1]) * (
                                    x[3 * k + 2] - x[3 * j + 2]) * (self.radii[k] + self.radii[i]) ** self.p / (
                                                (x[3 * k] - x[3 * i]) ** 2 + (
                                                x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                                        x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 4) / 2)
                            sum2_3_3 += -self.p * (self.p + 2) * (x[3 * k + 2] - x[3 * j + 2]) ** 2 * (
                                    self.radii[k] + self.radii[i]) ** self.p / ((x[3 * k] - x[3 * i]) ** 2 + (
                                    x[3 * k + 1] - x[3 * i + 1]) ** 2 + (x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 4) / 2)

                        sum1_1_1 += sum1_id_id
                        sum1_2_2 += sum1_id_id
                        sum1_3_3 += sum1_id_id
                        sum2_1_1 += sum2_id_id
                        sum2_2_2 += sum2_id_id
                        sum2_3_3 += sum2_id_id
                    else:
                        for j in range(self.center):
                            sum1_id_id += (1 - self.p) * ((self.radii[k] + self.radii[j]) ** self.p / (
                                    (x[3 * k] - x[3 * j]) ** 2 + (x[3 * k + 1] - x[3 * j + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * j + 2]) ** 2) ** ((self.p + 1) / 2))
                            sum1_1_1 += -(1 - self.p) * (self.p + 1) * (x[3 * k] - x[3 * j]) ** 2 * (
                                    self.radii[k] + self.radii[i]) ** self.p / ((x[3 * k] - x[3 * i]) ** 2 + (
                                    x[3 * k + 1] - x[3 * i + 1]) ** 2 + (x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 3) / 2)
                            sum1_1_2 += -(1 - self.p) * (self.p + 1) * (x[3 * k] - x[3 * j]) * (
                                    x[3 * k + 1] - x[3 * j + 1]) * (self.radii[k] + self.radii[i]) ** self.p / (
                                                (x[3 * k] - x[3 * i]) ** 2 + (
                                                x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                                        x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 3) / 2)
                            sum1_1_3 += -(1 - self.p) * (self.p + 1) * (x[3 * k] - x[3 * j]) * (
                                    x[3 * k + 2] - x[3 * j + 2]) * (self.radii[k] + self.radii[i]) ** self.p / (
                                                (x[3 * k] - x[3 * i]) ** 2 + (
                                                x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                                        x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 3) / 2)
                            sum1_2_2 += -(1 - self.p) * (self.p + 1) * (x[3 * k + 1] - x[3 * j + 1]) ** 2 * (
                                    self.radii[k] + self.radii[i]) ** self.p / ((x[3 * k] - x[3 * i]) ** 2 + (
                                    x[3 * k + 1] - x[3 * i + 1]) ** 2 + (x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 3) / 2)
                            sum1_2_3 += -(1 - self.p) * (self.p + 1) * (x[3 * k + 1] - x[3 * j + 1]) * (
                                    x[3 * k + 2] - x[3 * j + 2]) * (self.radii[k] + self.radii[i]) ** self.p / (
                                                (x[3 * k] - x[3 * i]) ** 2 + (
                                                x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                                        x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 3) / 2)
                            sum1_3_3 += -(1 - self.p) * (self.p + 1) * (x[3 * k + 2] - x[3 * j + 2]) ** 2 * (
                                    self.radii[k] + self.radii[i]) ** self.p / ((x[3 * k] - x[3 * i]) ** 2 + (
                                    x[3 * k + 1] - x[3 * i + 1]) ** 2 + (x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 3) / 2)
                            sum2_id_id += -self.p * ((self.radii[k] + self.radii[j]) ** self.p / (
                                    (x[3 * k] - x[3 * j]) ** 2 + (x[3 * k + 1] - x[3 * j + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * j + 2]) ** 2) ** ((self.p + 2) / 2))
                            sum2_1_1 += self.p * (self.p + 2) * (x[3 * k] - x[3 * j]) ** 2 * (
                                    self.radii[k] + self.radii[i]) ** self.p / ((x[3 * k] - x[3 * i]) ** 2 + (
                                    x[3 * k + 1] - x[3 * i + 1]) ** 2 + (x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 4) / 2)
                            sum2_1_2 += self.p * (self.p + 2) * (x[3 * k] - x[3 * j]) * (
                                    x[3 * k + 1] - x[3 * j + 1]) * (self.radii[k] + self.radii[i]) ** self.p / (
                                                (x[3 * k] - x[3 * i]) ** 2 + (
                                                x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                                        x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 4) / 2)
                            sum2_1_3 += self.p * (self.p + 2) * (x[3 * k] - x[3 * j]) * (
                                    x[3 * k + 2] - x[3 * j + 2]) * (self.radii[k] + self.radii[i]) ** self.p / (
                                                (x[3 * k] - x[3 * i]) ** 2 + (
                                                x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                                        x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 4) / 2)
                            sum2_2_2 += self.p * (self.p + 2) * (x[3 * k + 1] - x[3 * j + 1]) ** 2 * (
                                    self.radii[k] + self.radii[i]) ** self.p / ((x[3 * k] - x[3 * i]) ** 2 + (
                                    x[3 * k + 1] - x[3 * i + 1]) ** 2 + (x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 4) / 2)
                            sum2_2_3 += self.p * (self.p + 2) * (x[3 * k + 1] - x[3 * j + 1]) * (
                                    x[3 * k + 2] - x[3 * j + 2]) * (self.radii[k] + self.radii[i]) ** self.p / (
                                                (x[3 * k] - x[3 * i]) ** 2 + (
                                                x[3 * k + 1] - x[3 * i + 1]) ** 2 + (
                                                        x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 4) / 2)
                            sum2_3_3 += self.p * (self.p + 2) * (x[3 * k + 2] - x[3 * j + 2]) ** 2 * (
                                    self.radii[k] + self.radii[i]) ** self.p / ((x[3 * k] - x[3 * i]) ** 2 + (
                                    x[3 * k + 1] - x[3 * i + 1]) ** 2 + (x[3 * k + 2] - x[3 * i + 2]) ** 2) ** (
                                                (self.p + 4) / 2)
                        sum1_1_1 += sum1_id_id
                        sum1_2_2 += sum1_id_id
                        sum1_3_3 += sum1_id_id
                        sum2_1_1 += sum2_id_id
                        sum2_2_2 += sum2_id_id
                        sum2_3_3 += sum2_id_id
                else:
                    if k >= self.center:
                        if p < self.center:
                            sum1_1_1 = -(1 - self.p) * ((self.radii[k] + self.radii[p]) ** self.p / (
                                    (x[3 * k] - x[3 * p]) ** 2 + (x[3 * k + 1] - x[3 * p + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * p + 2]) ** 2) ** ((self.p + 1) / 2)) + (
                                               1 - self.p) * (self.p + 1) * (x[3 * k] - x[3 * p]) ** 2 * (
                                               self.radii[k] + self.radii[p]) ** self.p / (
                                               (x[3 * k] - x[3 * p]) ** 2 + (
                                               x[3 * k + 1] - x[3 * p + 1]) ** 2 + (
                                                       x[3 * k + 2] - x[3 * p + 2]) ** 2) ** (
                                               (self.p + 3) / 2)
                            sum1_1_2 = (1 - self.p) * (self.p + 1) * (x[3 * k] - x[3 * p]) * (
                                    x[3 * k + 1] - x[3 * p + 1]) * (self.radii[k] + self.radii[p]) ** self.p / (
                                               (x[3 * k] - x[3 * p]) ** 2 + (
                                               x[3 * k + 1] - x[3 * p + 1]) ** 2 + (
                                                       x[3 * k + 2] - x[3 * p + 2]) ** 2) ** (
                                               (self.p + 3) / 2)
                            sum1_1_3 = (1 - self.p) * (self.p + 1) * (x[3 * k] - x[3 * p]) * (
                                    x[3 * k + 2] - x[3 * p + 2]) * (self.radii[k] + self.radii[p]) ** self.p / (
                                               (x[3 * k] - x[3 * p]) ** 2 + (
                                               x[3 * k + 1] - x[3 * p + 1]) ** 2 + (
                                                       x[3 * k + 2] - x[3 * p + 2]) ** 2) ** (
                                               (self.p + 3) / 2)
                            sum1_2_2 = -(1 - self.p) * ((self.radii[k] + self.radii[p]) ** self.p / (
                                    (x[3 * k] - x[3 * p]) ** 2 + (x[3 * k + 1] - x[3 * p + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * p + 2]) ** 2) ** ((self.p + 1) / 2)) + (
                                               1 - self.p) * (self.p + 1) * (
                                               x[3 * k + 1] - x[3 * p + 1]) ** 2 * (
                                               self.radii[k] + self.radii[p]) ** self.p / (
                                               (x[3 * k] - x[3 * p]) ** 2 + (
                                               x[3 * k + 1] - x[3 * p + 1]) ** 2 + (
                                                       x[3 * k + 2] - x[3 * p + 2]) ** 2) ** (
                                               (self.p + 3) / 2)
                            sum1_2_3 = (1 - self.p) * (self.p + 1) * (x[3 * k + 1] - x[3 * p + 1]) * (
                                    x[3 * k + 2] - x[3 * p + 2]) * (self.radii[k] + self.radii[p]) ** self.p / (
                                               (x[3 * k] - x[3 * p]) ** 2 + (
                                               x[3 * k + 1] - x[3 * p + 1]) ** 2 + (
                                                       x[3 * k + 2] - x[3 * p + 2]) ** 2) ** (
                                               (self.p + 3) / 2)
                            sum1_3_3 = -(1 - self.p) * ((self.radii[k] + self.radii[p]) ** self.p / (
                                    (x[3 * k] - x[3 * p]) ** 2 + (x[3 * k + 1] - x[3 * p + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * p + 2]) ** 2) ** ((self.p + 1) / 2)) + (
                                               1 - self.p) * (self.p + 1) * (
                                               x[3 * k + 2] - x[3 * p + 2]) ** 2 * (
                                               self.radii[k] + self.radii[p]) ** self.p / (
                                               (x[3 * k] - x[3 * p]) ** 2 + (
                                               x[3 * k + 1] - x[3 * p + 1]) ** 2 + (
                                                       x[3 * k + 2] - x[3 * p + 2]) ** 2) ** (
                                               (self.p + 3) / 2)
                            sum2_1_1 = self.p * ((self.radii[k] + self.radii[p]) ** self.p / (
                                    (x[3 * k] - x[3 * p]) ** 2 + (x[3 * k + 1] - x[3 * p + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * p + 2]) ** 2) ** ((self.p + 2) / 2)) - self.p * (
                                               self.p + 2) * (x[3 * k] - x[3 * p]) ** 2 * (
                                               self.radii[k] + self.radii[p]) ** self.p / (
                                               (x[3 * k] - x[3 * p]) ** 2 + (
                                               x[3 * k + 1] - x[3 * p + 1]) ** 2 + (
                                                       x[3 * k + 2] - x[3 * p + 2]) ** 2) ** (
                                               (self.p + 4) / 2)
                            sum2_1_2 = -self.p * (self.p + 2) * (x[3 * k] - x[3 * p]) * (
                                    x[3 * k + 1] - x[3 * p + 1]) * (self.radii[k] + self.radii[p]) ** self.p / (
                                               (x[3 * k] - x[3 * p]) ** 2 + (
                                               x[3 * k + 1] - x[3 * p + 1]) ** 2 + (
                                                       x[3 * k + 2] - x[3 * p + 2]) ** 2) ** (
                                               (self.p + 4) / 2)
                            sum2_1_3 = -self.p * (self.p + 2) * (x[3 * k] - x[3 * p]) * (
                                    x[3 * k + 2] - x[3 * p + 2]) * (self.radii[k] + self.radii[p]) ** self.p / (
                                               (x[3 * k] - x[3 * p]) ** 2 + (
                                               x[3 * k + 1] - x[3 * p + 1]) ** 2 + (
                                                       x[3 * k + 2] - x[3 * p + 2]) ** 2) ** (
                                               (self.p + 4) / 2)
                            sum2_2_2 = self.p * ((self.radii[k] + self.radii[p]) ** self.p / (
                                    (x[3 * k] - x[3 * p]) ** 2 + (x[3 * k + 1] - x[3 * p + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * p + 2]) ** 2) ** ((self.p + 2) / 2)) - self.p * (
                                               self.p + 2) * (x[3 * k + 1] - x[3 * p + 1]) ** 2 * (
                                               self.radii[k] + self.radii[p]) ** self.p / (
                                               (x[3 * k] - x[3 * p]) ** 2 + (
                                               x[3 * k + 1] - x[3 * p + 1]) ** 2 + (
                                                       x[3 * k + 2] - x[3 * p + 2]) ** 2) ** (
                                               (self.p + 4) / 2)
                            sum2_2_3 = -self.p * (self.p + 2) * (x[3 * k + 1] - x[3 * p + 1]) * (
                                    x[3 * k + 2] - x[3 * p + 2]) * (self.radii[k] + self.radii[p]) ** self.p / (
                                               (x[3 * k] - x[3 * p]) ** 2 + (
                                               x[3 * k + 1] - x[3 * p + 1]) ** 2 + (
                                                       x[3 * k + 2] - x[3 * p + 2]) ** 2) ** (
                                               (self.p + 4) / 2)
                            sum2_3_3 = self.p * ((self.radii[k] + self.radii[p]) ** self.p / (
                                    (x[3 * k] - x[3 * p]) ** 2 + (x[3 * k + 1] - x[3 * p + 1]) ** 2 + (
                                    x[3 * k + 2] - x[3 * p + 2]) ** 2) ** ((self.p + 1) / 2)) - self.p * (
                                               self.p + 2) * (x[3 * k + 2] - x[3 * p + 2]) ** 2 * (
                                               self.radii[k] + self.radii[p]) ** self.p / (
                                               (x[3 * k] - x[3 * p]) ** 2 + (
                                               x[3 * k + 1] - x[3 * p + 1]) ** 2 + (
                                                       x[3 * k + 2] - x[3 * p + 2]) ** 2) ** (
                                               (self.p + 4) / 2)

                sum1_2_1 = sum1_1_2
                sum1_3_1 = sum1_1_3
                sum1_3_2 = sum1_2_3
                sum2_2_1 = sum2_1_2
                sum2_3_1 = sum2_1_3
                sum2_3_2 = sum2_2_3

                hess[3 * k, 3 * p] = self.alpha * (sum2 ** 2 * (
                        sum1_1_1 * sum2 - sum2_1 * sum1_1 - sum2_1_1 * sum1 - sum2_1 * sum1_1) + 2 * sum2 * sum2_1 * sum2_1 * sum1) / sum2 ** 4
                hess[3 * p, 3 * k] = hess[3 * k, 3 * p]
                hess[3 * k + 1, 3 * p] = self.alpha * (sum2 ** 2 * (
                        sum1_2_1 * sum2 - sum2_1 * sum1_2 - sum2_2_1 * sum1 - sum2_2 * sum1_1) + 2 * sum2 * sum2_1 * sum2_2 * sum1) / sum2 ** 4
                hess[3 * p, 3 * k + 1] = hess[3 * k + 1, 3 * p]
                hess[3 * k + 2, 3 * p] = self.alpha * (sum2 ** 2 * (
                        sum1_3_1 * sum2 - sum2_1 * sum1_3 - sum2_3_1 * sum1 - sum2_3 * sum1_1) + 2 * sum2 * sum2_1 * sum2_3 * sum1) / sum2 ** 4
                hess[3 * p, 3 * k + 2] = hess[3 * k + 2, 3 * p]
                hess[3 * k, 3 * p + 1] = self.alpha * (sum2 ** 2 * (
                        sum1_1_2 * sum2 - sum2_2 * sum1_1 - sum2_1_2 * sum1 - sum2_1 * sum1_2) + 2 * sum2 * sum2_2 * sum2_1 * sum1) / sum2 ** 4
                hess[3 * p + 1, 3 * k] = hess[3 * k, 3 * p + 1]
                hess[3 * k + 1, 3 * p + 1] = self.alpha * (sum2 ** 2 * (
                        sum1_2_2 * sum2 - sum2_2 * sum1_2 - sum2_2_2 * sum1 - sum2_2 * sum1_2) + 2 * sum2 * sum2_2 * sum2_2 * sum1) / sum2 ** 4
                hess[3 * p + 1, 3 * k + 1] = hess[3 * k + 1, 3 * p + 1]
                hess[3 * k + 2, 3 * p + 1] = self.alpha * (sum2 ** 2 * (
                        sum1_3_2 * sum2 - sum2_2 * sum1_3 - sum2_3_2 * sum1 - sum2_3 * sum1_2) + 2 * sum2 * sum2_2 * sum2_3 * sum1) / sum2 ** 4
                hess[3 * p + 1, 3 * k + 2] = hess[3 * k + 2, 3 * p + 1]
                hess[3 * k, 3 * p + 2] = self.alpha * (sum2 ** 2 * (
                        sum1_1_3 * sum2 - sum2_3 * sum1_1 - sum2_1_3 * sum1 - sum2_1 * sum1_3) + 2 * sum2 * sum2_3 * sum2_1 * sum1) / sum2 ** 4
                hess[3 * p + 2, 3 * k] = hess[3 * k, 3 * p + 2]
                hess[3 * k + 1, 3 * p + 2] = self.alpha * (sum2 ** 2 * (
                        sum1_2_3 * sum2 - sum2_3 * sum1_2 - sum2_2_3 * sum1 - sum2_2 * sum1_3) + 2 * sum2 * sum2_3 * sum2_2 * sum1) / sum2 ** 4
                hess[3 * p + 2, 3 * k + 1] = hess[3 * k + 1, 3 * p + 2]
                hess[3 * k + 2, 3 * p + 2] = self.alpha * (sum2 ** 2 * (
                        sum1_3_3 * sum2 - sum2_3 * sum1_3 - sum2_3_3 * sum1 - sum2_3 * sum1_3) + 2 * sum2 * sum2_3 * sum2_3 * sum1) / sum2 ** 4
                hess[3 * p + 2, 3 * k + 2] = hess[3 * k + 2, 3 * p + 2]

            grad[k] = self.alpha * (sum1_1 * sum2 - sum2_1 * sum1) / sum2 ** 2
            grad[k + 1] = self.alpha * (sum1_2 * sum2 - sum2_2 * sum1) / sum2 ** 2
            grad[k + 2] = self.alpha * (sum1_3 * sum2 - sum2_3 * sum1) / sum2 ** 2

        return value, grad, hess

