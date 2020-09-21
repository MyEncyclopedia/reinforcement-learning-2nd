import numpy as np


def matplot_bar3d_ex(V, title=''):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    # setup the figure and axes
    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig)

    _x = np.arange(4)
    _y = np.arange(4)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    Z = np.array([x for x in V])
    bottom = np.zeros_like(Z)
    width = depth = 1

    colours = plt.cm.rainbow_r(Z / min(Z))
    ax.bar3d(x, y, bottom, width, depth, Z, shade=True, color=colours)
    ax.set_title(title)
    ax.set_zlim(min(Z), 1)
    ax.set_xticks(np.arange(0, 5))
    ax.set_yticks(np.arange(0, 5))

    colour_map = plt.cm.ScalarMappable(cmap=plt.cm.rainbow_r)
    colour_map.set_array(-Z)

    col_bar = plt.colorbar(colour_map).set_label('')

    plt.show()


if __name__ == "__main__":
    V = np.array([0.0, -14.0, -20.0, -22.0, -14.0, -18.0, -20.0, -20.0, -20.0, -20.0, -18.0, -14.0, -22.0, -20.0, -14.0, 0.0])
    matplot_bar3d_ex(V)
