import unittest
import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

from src.constrained_min import interior_pt
from examples import get_qp_problem, get_lp_problem
from src.utils import plot_qp_path_3d, plot_lp_path, plot_function_values

output_dir = os.path.join(project_root, "output")
os.makedirs(output_dir, exist_ok=True)


class TestConstrainedMin(unittest.TestCase):

    def test_qp(self):
        func, ineqs, A, b, x0 = get_qp_problem()
        x_star, path, objs = interior_pt(func, ineqs, A, b, x0)

        # Check constraints
        self.assertAlmostEqual(np.sum(x_star), 1.0, places=5)
        for c in ineqs:
            self.assertGreater(c(x_star), 0)

        # Save plots
        qp_path_file = os.path.join(output_dir, "qp_path_3d.png")
        qp_obj_file = os.path.join(output_dir, "qp_objective_values.png")
        plot_qp_path_3d(path, title="Feasible Region + Central Path in 3D (QP)", save_path=qp_path_file)
        plot_function_values([objs], ["QP"], title="Objective vs Iteration (QP)", save_path=qp_obj_file)

        # Print final results
        print("QP Final x:", np.round(x_star, 2))
        print("QP Objective:", np.round(func(x_star), 2))
        print("QP Constraint sum(x):", np.round(np.sum(x_star)))
        print("QP Inequality constraint values:",  np.round([c(x_star) for c in ineqs], 2))


    def test_lp(self):
        func, ineqs, A, b, x0 = get_lp_problem()
        x_star, path, objs = interior_pt(func, ineqs, A, b, x0)

        # Check constraints
        for c in ineqs:
            self.assertGreater(c(x_star), 0)

        # Save plots
        lp_path_file = os.path.join(output_dir, "lp_path_2d.png")
        lp_obj_file = os.path.join(output_dir, "lp_objective_values.png")
        plot_lp_path(path, title="Feasible Region + Central Path (LP)", save_path=lp_path_file)
        plot_function_values([objs], ["LP"], title="Objective vs Iteration (LP)", save_path=lp_obj_file)

        # Print final results
        print("LP Final x:", np.round(x_star, 2))
        print("LP Objective:", np.round(-func(x_star), 2))  # original was max(x + y)
        print("LP Constraints values at final x:", np.round([c(x_star) for c in ineqs], 2))



if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

