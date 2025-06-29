import numpy as np
from scipy.optimize import minimize

import numpy as np
from scipy.optimize import minimize

def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0,
                t0=1.0, mu=10.0, tol=1e-8, max_iter=50):
    x = np.array(x0, dtype=float)
    t = t0
    m = len(ineq_constraints)
    path = [x.copy()]
    obj_values = []

    # Make sure x0 is strictly feasible for all inequality constraints
    for g in ineq_constraints:
        if g(x) <= 0:
            raise ValueError("Initial point must strictly satisfy all inequality constraints")

    def barrier_obj(x, t):
        penalty = 0.0
        for g in ineq_constraints:
            val = g(x)
            if val <= 0:
                return 1e10  # use large penalty instead of inf to keep optimizer numerically stable
            penalty -= np.log(val)
        return t * func(x) + penalty

    for _ in range(max_iter):
        constraints = []
        if eq_constraints_mat is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x, A=eq_constraints_mat, b=eq_constraints_rhs: A @ x - b
            })

        res = minimize(barrier_obj, x, args=(t,),
                       method='trust-constr',  # trust-constr handles equality constraints
                       constraints=constraints,
                       jac='2-point', options={'disp': False})

        if not res.success:
            print("Warning: inner minimization did not converge:", res.message)

        x = res.x
        path.append(x.copy())
        obj_values.append(func(x))

        # Stopping condition
        if m / t < tol:
            break
        t *= mu

    return x, np.array(path), obj_values

