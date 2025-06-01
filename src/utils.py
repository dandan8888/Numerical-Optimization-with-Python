import numpy as np
import matplotlib.pyplot as plt

def plot_contours(f, xlim, ylim, title="", paths=None, labels=None, save_path=None):


    x = np.linspace(xlim[0], xlim[1], 400)
    y = np.linspace(ylim[0], ylim[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f(np.array([x_, y_]))[0] for x_, y_ in zip(x_row, y_row)]
                  for x_row, y_row in zip(X, Y)])

    plt.figure()
    CS = plt.contour(X, Y, Z, levels=50)

    if paths:
        for path, label in zip(paths, labels):
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], label=label, marker='o')

    if labels:
        plt.legend()

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.grid(False)

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.tight_layout(pad=0)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  

    plt.show()
    plt.close()



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


