from enum import Enum

import numpy as np

import ildars.math_utils as util

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
        "CLOSEST_LINES_EXTENDED",
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
        elif loc_algo is LocalizationAlgorithm.CLOSEST_LINES:
            distance = distance_closest_lines(n, w, v)
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
                STR_ORIGINAL: ref_sig.direct_signal.original_sender_position,
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
            "map to normal algorithm",
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
    if minor != 0:
        upper = np.multiply(2, np.multiply(np.dot(n, n), np.dot(w, b)))
        p = np.divide(upper, minor)
    else:
        print(
            "warning: encountered minor of 0 when using",
            "reflection geometry",
        )
    return p


def distance_closest_lines(n: np.array, w: np.array, v: np.array) -> float:
    a = -2 * n
    w_inv = w - 2 * np.dot(w, n) * n
    u = np.cross(v, w_inv)
    # we only return lambda which by construction denotes the distance of the
    # sender
    return -(np.dot(np.cross(w_inv, a), u) / np.dot(np.cross(w_inv, v), u))


# Since closest lines extended operates on multiple reflection clusters, we
# need a new function for it
def compute_sender_positions_closest_lines_extended(clusters):
    # Collect unique list of direct signals. We will later add the computed
    # sender position for each direct signal.
    direct_signals = {}
    for cluster in clusters:
        for rs in cluster.reflected_signals:
            ds = rs.direct_signal
            if ds not in direct_signals:
                direct_signals[ds] = {rs: cluster.wall_normal}
            else:
                direct_signals[ds][rs] = cluster.wall_normal
    # now compute the sender position for each direct signal
    solution = []
    for ds in direct_signals:
        lines = [(np.array([0, 0, 0]), ds.direction)]
        for rs in direct_signals[ds]:
            n = direct_signals[ds][rs]
            u = util.normalize(n)
            w_inv = rs.direction - 2 * np.dot(rs.direction, u) * u
            lines.append((2 * n, w_inv))
        center_point = compute_closest_point(lines)
        solution.append(
            {
                STR_COMPUTED: center_point,
                STR_DIRECT_SIGNAL: ds,
                STR_ORIGINAL: ds.original_sender_position,
            }
        )
    return solution


# Given a list of lines with the form (p, d) where p is a point on the line and
# d is a vector parallel to the line, compute the closest point between all
# lines. Implementation is based on "Neares approaches to multiple lines in
# n-dimensional space" by Han and Bancroft, CREWE Research Report vol. 2, 2010
def compute_closest_point(lines):
    m = len(lines)
    G = []
    d = []
    for i in range(m):
        for j in range(3):
            row = np.zeros(3 + m)
            row[j] = 1
            row[i + 3] = -lines[i][1][j]
            G.append(row)
            d.append(lines[i][0][j])
    G = np.array(G)
    d = np.array(d)
    # We assume G to be invertible in general position
    assert np.linalg.matrix_rank(G) == G.shape[1]
    # Solve using least squares approach
    return np.array(
        list(np.matmul(np.matmul(np.linalg.inv(np.matmul(G.T, G)), G.T), d))[
            0:3
        ]
    )
