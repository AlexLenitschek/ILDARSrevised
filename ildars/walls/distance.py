import numpy as np

import ildars.math_utils as util


def compute_distance(direction, reflection_cluster):
    return sum(
        [
            compute_distance_from_measurement(direction, reflection)
            for reflection in reflection_cluster.reflected_signals
        ]
    ) / len(reflection_cluster.reflected_signals)


def compute_distance_from_measurement(direction, reflection):
    # Compute distance to sender using wall direction formula
    u = util.normalize(direction)
    v = reflection.direct_signal.direction
    w = reflection.direction
    delta = reflection.delta
    b = util.normalize(np.cross(np.cross(u, v), u))
    p = 0
    dot = np.dot(np.subtract(v, w), b)
    if dot != 0:
        p = np.abs(np.divide(np.multiply(np.dot(w, b), delta), dot))
    else:
        print("encountered parallel vectors in distance averaging")
    # now compute wall distance by projecting (r+s)/2 onto u
    r = w * (p + delta)
    s = v * p
    rshalf = np.divide(np.add(r, s), 2)
    wall_distance = np.abs(np.dot(rshalf, u))
    return wall_distance
