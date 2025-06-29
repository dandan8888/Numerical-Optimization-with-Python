import numpy as np
import matplotlib.pyplot as plt
from src.constrained_min import interior_pt
from src.utils import plot_qp_path_3d, plot_lp_path, plot_function_values


def get_qp_problem():
    func = lambda x: x[0]**2 + x[1]**2 + (x[2]+1)**2
    ineqs = [lambda x, i=i: x[i] for i in range(3)]
    A = np.array([[1, 1, 1]])
    b = np.array([1])
    x0 = np.array([0.1, 0.2, 0.7])
    return func, ineqs, A, b, x0

def get_lp_problem():
    func = lambda x: -1 * (x[0] + x[1])
    ineqs = [
        lambda x: x[1] + x[0] - 1,  
        lambda x: 1 - x[1],        
        lambda x: 2 - x[0],       
        lambda x: x[1]              
    ]
    return func, ineqs, None, None, np.array([0.5, 0.75])

