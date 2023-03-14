"""Functions for simulating error on previously generated measurements
"""

import numpy as np
from scipy.stats import vonmises_line, uniform
from scipy.spatial.transform import Rotation

import ildars.math_utils as util


def simulate_reflection_error(
    reflected_signals, von_mises_error, delta_error, wall_error
):
    for signal in reflected_signals:
        if von_mises_error > 0:
            signal.direction = simulate_directional_error(
                signal.direction, von_mises_error
            )
        if delta_error > 0:
            signal.delta = simulate_numeric_error(signal.delta, delta_error)
    if wall_error > 0:
        reflected_signals = simulate_wall_error(reflected_signals, wall_error)
    return reflected_signals


def simulate_directional_error(vector, von_mises_error):
    # Completely new function based on
    # https://math.stackexchange.com/questions/4343044/rotate-vector-by-a-random-little-amount
    # v = np.divide(vector, np.linalg.norm(vector))
    # Find orthonormal basis including

    orthogonal_vector = random_orthogonal_vector(vector)
    if von_mises_error > 0:
        random_angle = vonmises_line(von_mises_error).rvs()
    else:
        random_angle = uniform.rvs(-np.pi, 2 * np.pi)
    # rotate using the random orthogonal vector as rotation vector
    rotation = Rotation.from_rotvec(random_angle * orthogonal_vector)
    return rotation.apply(vector)


def random_orthogonal_vector(vector):
    normalized_vector = util.normalize(vector)
    rearranged_vector = np.array(
        [-1 * normalized_vector[2], normalized_vector[0], normalized_vector[1]]
    )
    tangent = np.cross(normalized_vector, rearranged_vector)
    bitangent = np.cross(normalized_vector, tangent)
    # use uniform distribution because scipy's vonmises does not support
    # concentration of 0. This should however be equivalent to the original
    # implementation in Mathmatica, see "Relationship to other distibutions" at
    # https://reference.wolfram.com/language/ref/VonMisesDistribution.html
    random_angle = uniform.rvs(-np.pi, 2 * np.pi)
    orth = np.add(
        np.multiply(tangent, np.sin(random_angle)),
        np.multiply(bitangent, np.cos(random_angle)),
    )
    return util.normalize(orth)


def simulate_numeric_error(delta, delta_error):
    return delta


def simulate_wall_error(reflected_signals, wall_error):
    return reflected_signals
