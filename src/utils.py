import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_function_values(f_values_list, labels, title="", save_path=None):
    plt.figure()
    for f_values, label in zip(f_values_list, labels):
        plt.plot(f_values, label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Function Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
    plt.close()


def plot_lp_path(path, title="Feasible Region + Central Path (LP)", save_path=None):
    path = np.array(path)

    # Manually define feasible region polygon
    polygon = np.array([
        [0, 1],
        [1, 0],
        [2, 0],
        [2, 1],
    ])

    plt.figure()
    plt.fill(polygon[:, 0], polygon[:, 1], color='lightgray', alpha=0.7, label='Feasible Region')
    plt.plot(path[:, 0], path[:, 1], marker='o', color='blue', label='Central Path')
    plt.scatter(path[-1, 0], path[-1, 1], color='red', zorder=5, label=f'Final Point: ({path[-1, 0]:.2f}, {path[-1, 1]:.2f})')

    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 1.5)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
    plt.close()


def plot_qp_path_3d(path, title="Feasible Region + Central Path in 3D (QP)", save_path=None):
    path = np.array(path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot feasible triangle (x + y + z = 1, x,y,z > 0)
    triangle = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    ax.add_collection3d(Poly3DCollection([triangle], alpha=0.3, color='lightgray', label='Feasible Region'))

    # Plot central path
    ax.plot(path[:, 0], path[:, 1], path[:, 2], marker='o', color='blue', label='Central Path')
    ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], color='red', label=f'Final Point: ({path[-1, 0]:.2f}, {path[-1, 1]:.2f}, {path[-1, 2]:.2f})')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
    plt.close()





