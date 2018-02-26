import numpy as np


class GradNorm:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, grad, **kwargs):
        return np.linalg.norm(grad) < self.threshold
