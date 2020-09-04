import numpy as np

def matplot_bar3d_ex(V):
    import numpy as np
    import matplotlib.pyplot as plt
    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    # setup the figure and axes
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # fake data
    _x = np.arange(4)
    _y = np.arange(4)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = [x + 22 for x in V]
    bottom = np.zeros_like(top)
    width = depth = 1

    ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
    ax1.set_title('Shaded')

    # ax2.bar3d(x, y, bottom, width, depth, top, shade=False)
    # ax2.set_title('Not Shaded')

    plt.show()


def plotly_example():
    import plotly_express as px
    import plotly
    import plotly.graph_objs as go
    plotly.offline.init_notebook_mode(connected=True)

    iris = px.data.iris()

    iris_plot = px.scatter(iris, x='sepal_width', y='sepal_length',
                           color='species', marginal_y='histogram',
                           marginal_x='box', trendline='ols')

    plotly.offline.plot(iris_plot)


def plot_v(V):
    import plotly_express as px
    import plotly
    import plotly.graph_objects as go
    plotly.offline.init_notebook_mode(connected=True)
    import numpy as np

    fig = go.Figure(go.Surface(
        # contours={
        #     "x": {"show": True, "start": 1.5, "end": 2, "size": 0.04, "color": "white"},
        #     "z": {"show": True, "start": 0.5, "end": 0.8, "size": 0.05}
        # },
        x=list(range(0, 4)),
        y=list(range(0, 4)),
        z=np.array(V).reshape(4, 4)
        # z=[
        #     [0, 1, 0, 1, 0],
        #     [1, 0, 1, 0, 1],
        #     [0, 1, 0, 1, 0],
        #     [1, 0, 1, 0, 1],
        #     [0, 1, 0, 1, 0]
        # ]
    ))
    # fig.update_layout(
    #     scene={
    #         "xaxis": {"nticks": 20},
    #         "zaxis": {"nticks": 4},
    #         'camera_eye': {"x": 0, "y": -1, "z": 0.5},
    #         "aspectratio": {"x": 1, "y": 1, "z": 0.2}
    #     })
    plotly.offline.plot(fig)
    # fig.show()


if __name__ == "__main__":
    V = np.array([0.0, -14.0, -20.0, -22.0, -14.0, -18.0, -20.0, -20.0, -20.0, -20.0, -18.0, -14.0, -22.0, -20.0, -14.0, 0.0])
    matplot_bar3d_ex(V)
