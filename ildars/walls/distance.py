import numpy as np


def compute_distance(direction, reflection_cluster):
    return sum(
        [
            compute_distance_from_measurement(direction, reflection)
            for reflection in reflection_cluster.reflected_signals
        ]
    ) / len(reflection_cluster.reflected_signals)


def compute_distance_from_measurement(direction, reflection):
    # Compute distance to sender using wall direction formula
    u = np.divide(direction, np.linalg.norm(direction))
    v = reflection.direct_signal.direction
    w = reflection.direction
    delta = reflection.delta
    b = np.cross(np.cross(u, v), u)
    p = 0
    dot = np.dot(np.subtract(v, w), b)
    if dot != 0:
        p = np.divide(np.multiply(np.dot(w, b), delta), dot)
    # now compute wall distance by projecting (r+s)/2 onto u
    r = np.multiply(v, p)
    s = np.multiply(w, np.add(p, delta))
    rshalf = np.divide(np.add(r, s), 2)
    return np.dot(rshalf, u)
