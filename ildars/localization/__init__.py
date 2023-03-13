from enum import Enum
import operator
import numpy as np

import ildars.localization.wall_selection as ws

WallSelectionMethod = Enum(
    "WallSelectionMethod",
    [
        "LARGEST_REFLECTION_CLUSTER",
        "NARROWEST_CLUSTER",
        "UNWEIGHTED_AVERAGE",
        "WEIGHTED_AVERAGE_WALL_DISTANCE",
        "CLOSEST_LINES_EXTENDED",
    ],
)
LocalizationAlgorithm = Enum(
    "LocalizationAlgorithm",
    [
        "MAP_TO_NORMAL_VECTOR",
        "CLOSEST_LINES",
        "REFLECTION_GEOMETRY",
        "WALL_DIRECTION",
    ],
)

_AveragingMethod = Enum("_AveragingMethod", ["EVEN"])


def compute_sender_positions(
    localization_algorithm,
    reflection_clusters,
    direct_signals,
    reflected_signals,
    algo=WallSelectionMethod.LARGEST_REFLECTION_CLUSTER,
):
    averaging_method = _AveragingMethod.EVEN
    cluster_selection = [reflection_clusters[0]]
    if algo is WallSelectionMethod.LARGEST_REFLECTION_CLUSTER:
        cluster_selection = ws.select_by_largest_cluster(reflection_clusters)
    # if algo is WallSelectionMethod.NARROWEST_CLUSTER:
    #     cluster
    else:
        raise NotImplementedError(
            "Wall selection algorithm",
            algo,
            "is either unknown or not implemented yet.",
        )
    return compute_sender_positions_largest_cluster(
        localization_algorithm, reflection_clusters, reflected_signals
    )


# Wall selection methods
def compute_sender_positions_largest_cluster(
    localization_algorithm, reflection_clusters, reflected_signals
):
    assert len(reflection_clusters) > 0
    largest_cluster = max(reflection_clusters, key=len)
    return compute_sender_positions_for_given_wall(
        localization_algorithm, largest_cluster.wall_normal, reflected_signals
    )


# Compute sender positions using closed formulae
def compute_sender_positions_for_given_wall(
    localization_algorithm, wall_nv, reflected_signals
):
    if localization_algorithm is LocalizationAlgorithm.WALL_DIRECTION:
        positions = []
        for ref_sig in reflected_signals:
            distance = distance_wall_direction(
                np.divide(wall_nv, np.linalg.norm(wall_nv)),
                ref_sig.direct_signal.direction,
                ref_sig.direction,
                ref_sig.delta,
            )
            positions.append(
                {
                    "computed": np.multiply(
                        ref_sig.direct_signal.direction, distance
                    ),
                    "original": ref_sig.original_sender_position,
                }
            )
        return positions

    else:
        raise NotImplementedError(
            "Localization algorithm",
            localization_algorithm,
            "is either unknown or not implemented yet",
        )


### Closed formulas for computing distance of sender p
# With given normalized vector v, u, w and the delta in m
def distance_wall_direction(u, v, w, delta):
    b = np.cross(np.cross(u, v), u)
    minor = np.dot(np.subtract(v, w), b)
    p = 0
    if minor != 0:
        p = np.divide(np.multiply(np.dot(w, b), np.abs(delta)), minor)
    return p


# With given normalized vectors v, w, the vector n to wall and the delta in m
def distance_map_to_normal(n, v, w, delta):
    minor = np.dot(np.add(v, w), n)
    p = 0
    if minor != 0:  # minor != 0
        upper = np.dot(
            np.subtract(np.multiply(2, n), np.multiply(np.abs(delta), w)), n
        )
        p = np.divide(upper, minor)
    return p


# With given normalized vectors u, v, w
def distance_reflection_geometry(u, n, v, w):
    b = np.cross(np.cross(u, v), u)
    minor = np.add(
        np.multiply(np.dot(v, n), np.dot(w, b)),
        np.multiply(np.dot(v, b), np.dot(w, n)),
    )
    p = 0
    if minor != 0:  # minor != 0
        upper = np.multiply(2, np.multiply(np.dot(n, n), np.dot(w, b)))
        p = np.divide(upper, minor)
    return p
