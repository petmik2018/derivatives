import numpy as np


class DiscreteFunction:

    def __init__(self, x, y):
        self.arguments = x
        self.values = y
        self.n = len(x) - 1
        self.h = (self.arguments[self.n] - self.arguments[0]) / self.n

    def numpy_diff(self):
        x_diff = np.zeros(self.n)
        y_diff = np.diff(self.values) / self.h
        for i in range(0, self.n):
            x_diff[i] = self.h / 2 + i * self.h
        return DiscreteFunction(x_diff, y_diff)

    def numpy_gradient(self):
        y_diff = np.gradient(self.values, self.h)
        return DiscreteFunction(self.arguments, y_diff)
