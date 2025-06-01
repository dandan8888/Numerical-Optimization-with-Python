import numpy as np

def quad1(x, hessian=False):
    Q = np.eye(2)
    f = x.T @ Q @ x
    g = 2 * Q @ x
    h = 2 * Q
    return (f, g, h) if hessian else (f, g, None)

def quad2(x, hessian=False):
    Q = np.diag([1, 100])
    f = x.T @ Q @ x
    g = 2 * Q @ x
    h = 2 * Q
    return (f, g, h) if hessian else (f, g, None)

def quad3(x, hessian=False):
    theta = np.pi / 6
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    Q = R.T @ np.diag([100, 1]) @ R
    f = x.T @ Q @ x
    g = 2 * Q @ x
    h = 2 * Q
    return (f, g, h) if hessian else (f, g, None)

def rosenbrock(x, hessian=False):
    f = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    g = np.array([-400*x[0]*(x[1] - x[0]**2) - 2*(1 - x[0]),
                  200*(x[1] - x[0]**2)])
    if hessian:
        h = np.array([[1200*x[0]**2 - 400*x[1] + 2, -400*x[0]],
                      [-400*x[0], 200]])
        return f, g, h
    return f, g, None

def linear(x, hessian=False):
    a = np.array([1.0, -2.0])
    f = a @ x
    g = a
    h = np.zeros((2, 2))
    return (f, g, h) if hessian else (f, g, None)

def smoothed_corner(x, hessian=False):
    f1 = np.exp(x[0] + 3*x[1] - 0.1)
    f2 = np.exp(x[0] - 3*x[1] - 0.1)
    f3 = np.exp(-x[0] - 0.1)
    f = f1 + f2 + f3
    g = np.array([
        f1 + f2 - f3,
        3*f1 - 3*f2
    ])
    if hessian:
        h = np.array([
            [f1 + f2 + f3, 3*f1 - 3*f2],
            [3*f1 - 3*f2, 9*f1 + 9*f2]
        ])
        return f, g, h
    return f, g, None


