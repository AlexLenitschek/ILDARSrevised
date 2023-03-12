from enum import Enum
import numpy as np

from .direction import (
    compute_direction_from_pairs,
    compute_direction_all_pairs_linear,
    STR_ALL,
    STR_DISJOINT,
    STR_OVERLAPPING,
)
from .distance import compute_distance

# Contains all methods for wall normal vector computation
# Functions take as input a set of measurements (here we use the
# ReflectedSignal class).
# It is assumend that all given measurements belong to the same wall.
WallNormalAlgorithm = Enum(
    "WallNormalAlgorithm",
    ["ALL_PAIRS", "LINEAR_ALL_PAIRS", "DISJOINT_PAIRS", "OVERLAPPING_PAIRS"],
)


def compute_wall_normal_vector(wall_normal_algorithm, reflection_cluster):
    if wall_normal_algorithm is WallNormalAlgorithm.ALL_PAIRS:
        compute_direction_from_pairs(reflection_cluster, STR_ALL)
    elif wall_normal_algorithm is WallNormalAlgorithm.LINEAR_ALL_PAIRS:
        compute_direction_all_pairs_linear(reflection_cluster)
    elif wall_normal_algorithm is WallNormalAlgorithm.OVERLAPPING_PAIRS:
        compute_direction_from_pairs(reflection_cluster, STR_OVERLAPPING)
    elif wall_normal_algorithm is WallNormalAlgorithm.DISJOINT_PAIRS:
        compute_direction_from_pairs(reflection_cluster, STR_DISJOINT)
    else:
        raise NotImplementedError(
            "Wall normal vector computation algorithm",
            wall_normal_algorithm,
            "is not known.",
        )
    direction = reflection_cluster.wall_normal
    distance = compute_distance(direction, reflection_cluster)
    reflection_cluster.wall_normal = np.multiply(direction, distance)
