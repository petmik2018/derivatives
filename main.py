import matplotlib.pyplot as plt
import numpy as np

from discreteFunction import DiscreteFunction

float_formatter = "{:.3f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})


def test_function(x):
    # return np.exp(x)
    return np.sin(x)


def test_derivative(x):
    return np.cos(x)


xMin = 0
# xMax = 1.0
xMax = np.pi / 2

n = 10
x = np.linspace(xMin, xMax, n+1)
y = test_function(x)
dydx = test_derivative(x)

sourceFunction = DiscreteFunction(x, y)
derivative_diff = sourceFunction.numpy_diff()
derivative_gradient = sourceFunction.numpy_gradient()

plt.xlim(0.0, 0.3)
plt.ylim(0.95, 1.05)
plt.plot(x, dydx, color='green', linewidth=2, label="test")
plt.plot(derivative_diff.arguments, derivative_diff.values, color='blue', linewidth=2, label="np.diff")
plt.plot(derivative_gradient.arguments, derivative_gradient.values, color='black', linewidth=2, label="gradient")
plt.legend()
plt.show()

