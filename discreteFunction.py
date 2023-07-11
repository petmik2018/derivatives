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

    def points_3(self, mode='valid'):
        kernel_3 = np.array([1 / 2, 0, -1 / 2])
        if mode == "same":
            x_diff = self.arguments
        else:
            x_diff = self.arguments[:-1][1:]
        y_diff = np.convolve(kernel_3, self.values, mode=mode) / self.h
        return DiscreteFunction(x_diff, y_diff)

    def points_5(self, mode='valid'):
        kernel_5 = np.array([-1 / 12, 8 / 12, 0, -8 / 12, 1 / 12])
        if mode == "same":
            x_diff = self.arguments
        else:
            x_diff = self.arguments[:-2][2:]
        y_diff = np.convolve(kernel_5, self.values, mode=mode) / self.h
        return DiscreteFunction(x_diff, y_diff)

    def approximation(self):

        def get_abcd(f, k: int, n: int, h: float):
            # a*x[i-1] + b*x[i] + c*x[i+1] = d
            if k == 0:
                return 0, h / 3, h / 6, - f[0] + (f[1] + f[0]) / 2
            if k == n:
                return h / 6, h / 3, 0, f[n] - (f[n] + f[n - 1]) / 2
            else:
                return h / 6, 2 * h / 3, h / 6, (f[k + 1] - f[k - 1]) / 2

        alfa = np.zeros(self.n)
        beta = np.zeros(self.n)
        approx = np.zeros(self.n + 1)

        a, b, c, d = get_abcd(self.values, 0, self.n, self.h)
        alfa[0] = - c / b
        beta[0] = d / b

        for i in range(1, self.n):
            a, b, c, d = get_abcd(self.values, i, self.n, self.h)
            alfa[i] = -c / (a * alfa[i - 1] + b)
            beta[i] = (d - a * beta[i - 1]) / (a * alfa[i - 1] + b)

        a, b, c, d = get_abcd(self.values, self.n, self.n, self.h)
        approx[self.n] = (d - a * beta[self.n - 1]) / (a * alfa[self.n - 1] + b)

        for k in range(self.n - 1, -1, -1):
            approx[k] = alfa[k] * approx[k + 1] + beta[k]

        return DiscreteFunction(self.arguments, approx)
