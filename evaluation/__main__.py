import random

import ildars
from . import testrooms
from . import signal_simulation
from .renderer import Renderer

# Error setup
VON_MISES_CONCENTRATION = 1  # concentration of 1 lead so uniform distribution
DELTA_ERROR = 0.1  # 10cm standard deviation
WALL_ERROR = 0  # no wrongly assigned reflections

receiver_position = (0.1, 0.2, 0.3)


def run_experiment():
    # random sender positions
    sender_positions = [
        (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
        for i in range(10)
    ]

    (
        direct_signals,
        reflected_signals,
    ) = signal_simulation.generate_measurements(
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
    return sender_positions, clusters, positions


pos_orig, clusters, pos_comp = run_experiment()


def new_experiment():
    pos_orig, clusters, pos_comp = run_experiment()
    return clusters, pos_orig, pos_comp


renderer = Renderer(
    receiver_position, new_experiment, clusters, pos_orig, pos_comp
)
