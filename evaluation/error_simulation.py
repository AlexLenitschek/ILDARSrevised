"""Functions for simulating error on previously generated measurements
"""

import numpy as np
from scipy.stats import vonmises_line, uniform, norm

import ildars.math_utils as util


def simulate_error(
    direct_signals, reflected_signals, von_mises_error, delta_error, wall_error
):
    for signal in direct_signals:
        if von_mises_error > 0:
            signal.direction = simulate_directional_error(
                signal.direction, von_mises_error
            )
    for signal in reflected_signals:
        if von_mises_error > 0:
            signal.direction = simulate_directional_error(
                signal.direction, von_mises_error
            )
        if delta_error > 0:
            signal.delta = simulate_numeric_error(signal.delta, delta_error)
    if wall_error > 0:
        reflected_signals = simulate_wall_error(
            reflected_signals, wall_error, direct_signals
        )
    return (direct_signals, reflected_signals)


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
    return np.abs(delta + norm.rvs(loc=0, scale=delta_error))


def simulate_wall_error(reflected_signals, wall_error, direct_signals):
    # select a random sample of the size implied by wall_error
    rng = np.random.default_rng()
    num_modified_reflections = int(
        np.round(len(reflected_signals) * wall_error)
    )
    modified_reflections = [
        reflected_signals[i]
        for i in rng.choice(
            a=len(reflected_signals),
            size=(num_modified_reflections),
            replace=False,
        )
    ]
    for ref in modified_reflections:
        # Choose a random direct signal, which is not the actual direct signal
        # of the current reflected signal
        new_direct_signal = direct_signals[rng.integers(len(direct_signals))]
        while new_direct_signal == ref.direct_signal:
            new_direct_signal = direct_signals[
                rng.integers(len(direct_signals))
            ]
        # now swap out the old reflected signal for the new one
        ref.direct_signal = new_direct_signal
    return reflected_signals
