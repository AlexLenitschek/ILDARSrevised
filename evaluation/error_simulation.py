"""Functions for simulating error on previously generated measurements
"""
import numpy as np
from scipy.stats import vonmises

def simulate_reflection_error(reflected_signals, von_mises_error, delta_error, wall_error):
    for signal in reflected_signals:
        if von_mises_error > 0:
            signal.direction = simulate_directional_error(signal.direction, von_mises_error)
        if delta_error > 0:
            signal.delta = simulate_numeric_error(signal.delta, delta_error)
    if wall_error > 0:
        reflected_signals = simulate_wall_error(reflected_signals, wall_error)
    return reflected_signals

def simulate_directional_error(vector, von_mises_error):
    orthogonal_vector = random_orthogonal_vector(vector)
    return vector

def random_orthogonal_vector(vector):
    normalized_vector = np.divide(vector, np.linalg.norm(vector))
    rearranged_vector = np.array([-1 * normalized_vector[2], normalized_vector[0], normalized_vector[1]])
    tangent = np.cross(normalized_vector, rearranged_vector)
    bitangent = np.cross(normalized_vector, tangent)
    random_angle = vonmises.rvs(1)
    return vector

def simulate_numeric_error(delta, delta_error):
    return delta

def simulate_wall_error(reflected_signals, wall_error):
    return reflected_signals