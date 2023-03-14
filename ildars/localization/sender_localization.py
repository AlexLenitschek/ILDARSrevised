from enum import Enum

import numpy as np

STR_COMPUTED = "computed"
STR_DIRECT_SIGNAL = "direct_signal"
STR_ORIGINAL = "original"

LocalizationAlgorithm = Enum(
    "LocalizationAlgorithm",
    [
        "MAP_TO_NORMAL_VECTOR",
        "CLOSEST_LINES",
        "REFLECTION_GEOMETRY",
        "WALL_DIRECTION",
    ],
)


# Compute sender positions using closed formulae, i.e.: every localization
# algorithm except for closest lines extended
def compute_sender_positions_for_given_wall(
    loc_algo, wall_nv, reflected_signals
):
    positions = []
    for ref_sig in reflected_signals:
        n = wall_nv
        u = np.divide(n, np.linalg.norm(n))
        v = ref_sig.direct_signal.direction
        w = ref_sig.direction
        delta = ref_sig.delta
        if loc_algo is LocalizationAlgorithm.WALL_DIRECTION:
            distance = distance_wall_direction(u, v, w, delta)
        elif loc_algo is LocalizationAlgorithm.MAP_TO_NORMAL_VECTOR:
            distance = distance_map_to_normal(n, v, w, delta)
        elif loc_algo is LocalizationAlgorithm.REFLECTION_GEOMETRY:
            distance = distance_reflection_geometry(u, n, v, w)
        else:
            raise NotImplementedError(
                "Localization algorithm",
                loc_algo,
                "is either unknown or not implemented yet",
            )

        positions.append(
            {
                STR_COMPUTED: np.multiply(
                    ref_sig.direct_signal.direction, distance
                ),
                STR_DIRECT_SIGNAL: ref_sig.direct_signal,
                STR_ORIGINAL: ref_sig.original_sender_position,
            }
        )
    return positions


# Closed formulas for computing distance of sender p
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
    if minor != 0:
        upper = np.dot(
            np.subtract(np.multiply(2, n), np.multiply(np.abs(delta), w)), n
        )
        p = np.divide(upper, minor)
    else:
        print(
            "warning: encountered minor of 0 when using",
            "distance to normal algorithm",
        )
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
