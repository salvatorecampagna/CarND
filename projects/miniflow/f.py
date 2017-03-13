import random
from gd import gradient_descent_update
import matplotlib.pyplot as plt


def f(x):
    """
    Quadratic function.

    It's easy to see the minimum value of the function
    is 5 when is x=0.
    """
    return x**2 + 5


def df(x):
    """
    Derivative of `f` with respect to `x`.
    """
    return 2*x


# Random number better 0 and 10,000. Feel free to set x whatever you like.
x = random.randint(0, 10000)
# TODO: Set the learning rate
learning_rate = 0.1
epochs = 100
costs = list()
for i in range(epochs+1):
    cost = f(x)
    gradx = df(x)
    costs.append(cost)
    print("EPOCH {}: Cost = {:.3f}, x = {:.3f}".format(i, cost, gradx))
    x = gradient_descent_update(x, gradx, learning_rate)

plt.plot(costs)
plt.show()