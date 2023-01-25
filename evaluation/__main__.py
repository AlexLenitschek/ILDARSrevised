import random
import os

import vedo

import ildars
from . import testrooms
from . import signal_simulation

# Error setup
VON_MISES_CONCENTRATION = 1  # concentration of 1 lead so uniform distribution
DELTA_ERROR = 0.1  # 10cm standard deviation
WALL_ERROR = 0  # no wrongly assigned reflections


receiver_position = (0.1, 0.1, 0.1)
# random sender positions
sender_positions = [
    (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
    for i in range(10)
]

(direct_signals, reflected_signals) = signal_simulation.generate_measurements(
    receiver_position, sender_positions, testrooms.CUBE
)
# TODO: fix and use error simulation
# reflected_signals = error_simulation.simulate_reflection_error(
#     reflected_signals, VON_MISES_CONCENTRATION, DELTA_ERROR, WALL_ERROR
# )

# For now: hardcoded inversion based clustering. In the future, evaulate
# all implemented algorithms of the ILDARS pipeline here
clusters, positions = ildars.run_ildars(
    direct_signals,
    reflected_signals,
    ildars.clustering.ClusteringAlgorithm.INVERSION,
    ildars.walls.WallNormalAlgorithm.ALL_PAIRS,
    ildars.localization.WallSelectionMethod.LARGEST_REFLECTION_CLUSTER,
    ildars.localization.LocalizationAlgorithm.WALL_DIRECTION,
)

# Plot room alongside the actual and computed sender positions
mesh_room = vedo.Mesh(
    os.getcwd() + "/evaluation/testrooms/models/cube.obj"
).wireframe()

mesh_positions_original = [
    vedo.Point(position_original, c="red")
    for position_original in sender_positions
]
mesh_positions_computed = [
    vedo.Point(position_computed, c="blue") for position_computed in positions
]
mesh_normals = [
    vedo.Arrow(
        start_pt=receiver_position,
        end_pt=(receiver_position + cluster.wall_normal),
        s=0.002,
        c="green",
    )
    for cluster in clusters
]
mesh_axes = vedo.Axes(
    mesh_room, xrange=(-1.5, 1.5), yrange=(-1.5, 1.5), zrange=(-1.5, 1.5)
)


# Functions for visualization
def show_clusters():
    print("showing clusters")


def show_normals():
    # hide other meshes
    for pos_o in mesh_positions_original:
        pos_o.alpha(0)
    for pos_c in mesh_positions_computed:
        pos_c.alpha(0)
    print("showing normals")


def show_positions():
    print("showing positions")


plt = vedo.Plotter()
plt.add_button(
    show_clusters, pos=(0.5, 0.1), states=("show reflection clusters", "n")
)
plt.add_button(
    show_normals, pos=(0.5, 0.05), states=("show wall normals", "n")
)
plt.add_button(
    show_positions, pos=(0.5, 0), states=("show sender positions", "n")
)
plt.show(
    mesh_room,
    mesh_positions_original,
    mesh_positions_computed,
    mesh_normals,
    mesh_axes,
    camera={"pos": (5, 5, 5), "viewup": (0, 0, 1), "focal_point": (0, 0, 0)},
).close()
