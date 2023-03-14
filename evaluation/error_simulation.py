"""Functions for simulating error on previously generated measurements
"""

import numpy as np
from scipy.stats import vonmises_line, uniform

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
    v = util.normalize(vector)
    # Find a random vector that is not parallel to v
    r = util.normalize(np.random.rand(3))
    while abs(np.dot(r, v)) == 1:
        r = util.normalize(np.random.rand(3))
    u1 = util.normalize(np.cross(v, r))
    u2 = util.normalize(np.cross(v, u1))
    # Now (v,u1,u2) are an orthonormal basis of R^3
    B = np.array([u1, u2, v])
    # Get random angle using von Mises distribution
    if von_mises_error > 0:
        theta = vonmises_line(von_mises_error).rvs()
    else:
        theta = uniform.rvs(-np.pi, np.pi)
    phi = uniform.rvs(-np.pi, np.pi)
    # Get rotated vector relative to B
    rotated_vector_b = np.dot(
        np.linalg.norm(vector),
        np.array(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ]
        ),
    )
    return B.T.dot(rotated_vector_b)


def simulate_numeric_error(delta, delta_error):
    return delta


def simulate_wall_error(reflected_signals, wall_error):
    return reflected_signals
