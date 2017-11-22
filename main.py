from pylab import *
# from scipy.optimize import minimize
from res.function import F


def evaluate_derivative(i, x0, f):
    n = len(x0)
    assert 0 <= i < n
    delta_x = 0.0001

    x = x0
    x_delted = list(x)
    x_delted[i] += delta_x

    result = (f(x_delted) - f(x)) / delta_x
    return result


def evaluate_gradient(f, x0):
    grad = [evaluate_derivative(i, x0, f) for i in range(len(x0))]
    return np.array(grad)


def evaluate_alpha(f, x):
    def minimizable(alpha):
        grad = evaluate_gradient(f, x)
        return f(x - alpha * grad)

    res = minimize(minimizable, np.array([0]))
    return res.x


def step(f, x):
    alpha = evaluate_alpha(f, x)
    grad = evaluate_gradient(f, x)
    return x - alpha*grad


def norm(x):
    return math.sqrt(np.sum(x**2))


def minimize_f(f, x0, eps):
    n = 10000
    i = 0
    dot = [np.array(x0), step(f, np.array(x0))]
    while i < n and norm(dot[-1] - dot[-2]) > eps:
        dot.append(step(f, dot[-1]))
        i += 1

    print(dot)
    return dot[-1]


def run():
    file_name = 'res/data.txt'

    # with open(file_name) as f:
    #     x0, eps, left, right = [float(x) for x in f.readline().split(' ')]  # read first line
    # print("point = ", x0, "eps = ", eps, "left = ", left, "right = ", right)
    #
    # points_num = int((right - left) / (eps / 2))
    # print('points_num = ', points_num)

    # x = np.linspace(left, right, points_num)
    # fx = F(x)
    print(evaluate_gradient(F, np.array([2.6])))
    print(minimize_f(F, [2.6], 0.001))
    pass


run()
