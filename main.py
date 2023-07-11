import matplotlib.pyplot as plt
import numpy as np

from discreteFunction import DiscreteFunction

float_formatter = "{:.3f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})


def test_function(x):  # function to be differentiated
    # return np.exp(x)
    return np.sin(x)


def test_derivative(x):  # analitic derivative of tested function
    # return np.exp(x)
    return np.cos(x)

# parameters of arguments set = array[0, ..., n+1]
xMin = 0
# xMax = 1.0
xMax = np.pi / 2
n = 20

x = np.linspace(xMin, xMax, n+1)
y = test_function(x)
dydx = test_derivative(x)

sourceFunction = DiscreteFunction(x, y)
derivative_diff = sourceFunction.numpy_diff()  # numpy.diff method
derivative_gradient = sourceFunction.numpy_gradient()  # numpy.gradient method
derivative_3_points = sourceFunction.points_3()  # 3-points approximation, possible parameter mode='same'
derivative_5_points = sourceFunction.points_5()  # 3-points approximation, possible parameter mode='same'
approx_fem = sourceFunction.approximation()  # finite elements method + congjugate approximation

plt.xlim(0.0, 0.2)  # window sizes for the plots
plt.ylim(0.977, 1.001)

plt.plot(derivative_diff.arguments, derivative_diff.values, color='blue', linewidth=2, label="np.diff")
plt.plot(derivative_gradient.arguments, derivative_gradient.values, color='black', linewidth=2, label="np.gradient")
plt.plot(derivative_3_points.arguments, derivative_3_points.values, color='yellow', linewidth=2, label="3-points")
plt.plot(derivative_5_points.arguments, derivative_5_points.values, color='violet', linewidth=2, label="5-points")
plt.plot(approx_fem.arguments, approx_fem.values, color='red', linewidth=2, label="finite elements")
plt.plot(x, dydx, color='green', linewidth=2, label="test")

# if you cannot see approx_fem plot, comment dydx plot
# if yu cannot see 5-points plot, comment approx_fem and dydx plots

plt.legend()
plt.show()

