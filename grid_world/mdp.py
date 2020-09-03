from enum import Enum
from typing import Dict, Tuple, List, Set


class Action(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

State = int
Reward = float
Prob = float
MDP = Dict[Tuple[State, Action], List[Tuple[State, Reward, Prob]]]
Policy = Dict[State, Dict[Action, Prob]]
Value = List[float]
StateSet = Set[int]
NonTerminalStateSet = Set[int]

def build_mdp() -> MDP:
    """

    :return:  (s, a) -> (s', r)
    """
    mdp_dict = {}
    for s in range(1, 4*4 - 1):
        r = s // 4
        c = s % 4
        for action in list(Action):
            neighbor_r = min(3, max(0, r + action.value[0]))
            neighbor_c = min(3, max(0, c + action.value[1]))
            s_ = neighbor_r * 4 + neighbor_c
            mdp_dict[(s, action)] = [(s_, -1, 1.0)]
    return mdp_dict

def random_policy() -> Policy:
    policy_dict = {}
    for s in range(0, 4*4):
        policy_dict[s] = {action: 0.25 for action in list(Action)}

    return policy_dict

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

    top = [x + 30 for x in V]
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
    mdp = build_mdp()
    # plot_v(V)
    # plotly_example()
    # matplot_bar3d_ex(V)
