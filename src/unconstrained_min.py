import numpy as np


def backtracking_line_search(f, x, p, grad, c1=0.01, beta=0.5, max_iter=20):
    alpha = 1.0
    for _ in range(max_iter):
        new_x = x + alpha * p
        f_new, _, _ = f(new_x)
        f_curr, _, _ = f(x)

        # Wolfe condition
        armijo = f_new <= f_curr + c1 * alpha * np.dot(grad, p)

        if armijo:
            return alpha

        alpha *= beta

    return alpha


def minimize(f, x0, method="gd", obj_tol=1e-12, param_tol=1e-8, max_iter=100):
    x = x0.copy()
    path = [x.copy()]
    f_values = []

    grad_f = lambda x: f(x, hessian=(method == "newton"))[1]

    for i in range(max_iter):
        fx, grad, hess = f(x, hessian=(method == "newton"))
        f_values.append(fx)
        print(f"Iter {i}: x = {x}, f(x) = {fx}")

        if method == "gd":
            p = -grad

        elif method == "newton":
            try:
                p = -np.linalg.solve(hess, grad)
                newton_decrement = np.sqrt(grad.T @ (-p))
                if newton_decrement < np.sqrt(2 * obj_tol):
                    print(f"Converged due to small Newton decrement: {newton_decrement:.6e}")
                    return x, fx, True, path, f_values
            except np.linalg.LinAlgError as e:
                print(f"LinAlgError in Newton step: {e}. Switching to gradient descent from now on.")
                method = 'gd'
                p = -grad

        alpha = backtracking_line_search(f, x, p, grad)
        x_new = x + alpha * p
        path.append(x_new.copy())

        if np.linalg.norm(x_new - x) < param_tol or abs(fx - f(x_new)[0]) < obj_tol:
            return x_new, f(x_new)[0], True, path, f_values

        x = x_new

    return x, f(x)[0], False, path, f_values

