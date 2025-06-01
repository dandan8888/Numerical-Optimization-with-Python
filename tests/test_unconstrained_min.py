import unittest
import numpy as np
import sys
import os
import matplotlib.pyplot as plt


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

from src.unconstrained_min import minimize
from src.utils import plot_contours, plot_function_values
import examples as examples

class TestUnconstrainedMin(unittest.TestCase):
    def test_all_examples(self):
        test_cases = {
            "Quadratic 1": (examples.quad1, [1.0, 1.0]),
            "Quadratic 2": (examples.quad2, [1.0, 1.0]),
            "Quadratic 3": (examples.quad3, [1.0, 1.0]),
            "Rosenbrock": (examples.rosenbrock, [-1.0, 2.0]),
            "Linear": (examples.linear, [1.0, 1.0]),
            "Smoothed Corner": (examples.smoothed_corner, [1.0, 1.0]),
        }

        output_dir = os.path.join(project_root, "output")
        os.makedirs(output_dir, exist_ok=True)

        for name, (f, x0) in test_cases.items():
            print(f"\nTesting: {name}")

            methods = ['gd', 'newton']
            paths = []
            f_vals = []
            labels = []

            for method in methods:
                max_iter = 10000 if (name == "Rosenbrock" and method == "gd") else 100
                x_final, f_final, success, path, f_values = minimize(
                    f, np.array(x0), method=method,
                    obj_tol=1e-12, param_tol=1e-8,
                    max_iter=max_iter
                )
                print(f"Final {method.upper()}: x = {x_final}, f = {f_final}, success = {success}")
                paths.append(path)
                f_vals.append(f_values)
                labels.append(method.upper())

            # Save plots
            contour_path = os.path.join(output_dir, f"{name.replace(' ', '_')}_contour.png")
            fvals_path = os.path.join(output_dir, f"{name.replace(' ', '_')}_fvals.png")

            plot_contours(f, [-2.5, 2.5], [-2.5, 2.5], title=name, paths=paths, labels=labels, save_path=contour_path)
            plot_function_values(f_vals, labels, title=name, save_path=fvals_path)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
