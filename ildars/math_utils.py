# "low-level" mathematical functions that are needed in ildars algorithms but
# are not included in any of the math packages (like numpy or scipy) we use

import numpy as np
import scipy as sp
#from scipy.linalg import norm as scipy_norm


# For a given vector, returns the parallel unit vector
def normalize(v: np.array) -> np.array:
    return v / np.linalg.norm(v)

# def normalize(v: np.array) -> np.array:
#     norm = np.linalg.norm(v)
#     if norm < 0.000001:
#         # Handle the case where the vector is a zero vector or very very small
#         return np.zeros_like(v)
#     return v / norm


# Get the relative angular distance between two vetors.
# 0 means the vectors are parallel, 2 means they are opposite.
# Since this function does not use triangular functions, it is more efficient
# compared to computing the angle between two vectors, but should only be used
# to compare two (or more) given vectors in terms of angular distance
def get_angular_dist(v1: np.array, v2: np.array) -> float:
    return abs(np.dot(normalize(v1), normalize(v2)) - 1)


# get the angle between two vectors. Implementation taken from
# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def get_angle(v1: np.array, v2: np.array) -> float:
    u1 = normalize(v1)
    u2 = normalize(v2)
    return np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))
