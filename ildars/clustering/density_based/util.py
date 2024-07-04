import sys
sys.path.append ('../ILDARSrevised')
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from enum import Enum
import toml
import ildars.math_utils as util
from ildars.clustering.cluster import ReflectionCluster
# Very small number
EPSILON = 0.000000001

def compute_distance_between_lines(line1, line2):
    closest_points = get_closest_points_on_lines(line1, line2)
    shortest_distance = np.linalg.norm(closest_points[0] - closest_points[1])
    return shortest_distance

# Helper function to convert lines to reflected signals
def lines_to_reflected_signals(lines):
    return [line.reflected_signal for line in lines]

# Helper functions taken from Milan Müller's Inversion Implementation
def invert_vector(vec):
    divisor = vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2
    return np.divide(vec, divisor)
    
# Based on first answer from
    # https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments
#@staticmethod
def get_closest_points_on_lines(line1, line2):
    pA = 0  # Closest point on line1 to line2
    pB = 0  # Closest point on line2 to line1
    cross = np.cross(line1.direction, line2.direction)
    denominator = np.linalg.norm(cross)
    if denominator < EPSILON:
         # Lines are parallel
        d0 = np.dot(line1.direction, np.subtract(line2.p1, line1.p1))
        d1 = np.dot(line1.direction, np.subtract(line2.p2, line1.p1))
        if d0 <= 0 >= d1:
            if np.absolute(d0) < np.absolute(d1):
                pA = line1.p1
                pB = line2.p1
            else:
                pA = line1.p1
                pB = line2.p2
        elif d0 >= np.linalg.norm(np.subtract(line1.p2, line1.p1)) <= d1:
            if np.absolute(d0) < np.absolute(d1):
                pA = line1.p2
                pB = line2.p1
            else:
                pA = line1.p2
                pB = line2.p2
        else:
            # Segments are parallel and overlapping.
            # No unique solution exists.
            center = np.divide(
                np.add(
                    np.add(line1.p1, line1.p2), np.add(line2.p1, line2.p2)
                ),
                4,
            )
            # compute projection on lines
            line1_to_center = np.subtract(center, line1.p1)
            line1_projection_length = np.dot(
                line1_to_center, line1.direction
            )
            line1_projection = np.add(
                line1.p1,
                np.multiply(line1.direction, line1_projection_length),
            )
            pA = line1_projection
            line2_to_center = np.subtract(center, line2.p1)
            line2_projection_length = np.dot(
                line2_to_center, line2.direction
            )
            line2_projection = np.add(
                line2.p1,
                np.multiply(line2.direction, line2_projection_length),
            )
            pB = line2_projection

    else:
        # Lines are not parallel
        t = np.subtract(line2.p1, line1.p1)

        detA = np.linalg.det([t, line2.direction, cross])
        detB = np.linalg.det([t, line1.direction, cross])

        t0 = detA / denominator
        t1 = detB / denominator

        # Compute projections
        pA = np.add(line1.p1, np.multiply(line1.direction, t0))
        pB = np.add(line2.p1, np.multiply(line2.direction, t1))

        # Clamp projections
        if t0 < 0:
            pA = line1.p1
        elif t0 > np.linalg.norm(np.subtract(line1.p2, line1.p1)):
            pA = line1.p2

        if t0 < 0:
            pB = line2.p1
        elif t0 > np.linalg.norm(np.subtract(line2.p2, line2.p1)):
            pB = line2.p2

        if t0 < 0 or t0 > np.linalg.norm(np.subtract(line1.p2, line1.p1)):
            dot = np.dot(line2.direction, np.subtract(pA, line2.p1))
            if dot < 0:
                dot = 0
            elif dot > np.linalg.norm(np.subtract(line2.p2, line2.p1)):
                dot = np.linalg.norm(np.subtract(line2.p2, line2.p1))
            pB = line2.p1 + np.multiply(line2.direction, dot)

        if t1 < 0 or t1 > np.linalg.norm(np.subtract(line2.p2, line2.p1)):
            dot = np.dot(line1.direction, np.subtract(pB, line1.p1))
            if dot < 0:
                dot = 0
            elif dot > np.linalg.norm(np.subtract(line1.p2, line1.p1)):
                dot = np.linalg.norm(np.subtract(line1.p2, line1.p1))
            pA = line1.p1 + np.multiply(line1.direction, dot)
    return (pA, pB)

# Classes
# Dataclass
class Segment:
    def __init__(self, p1, p2, reflected_signal):
        self.p1 = p1
        self.p2 = p2
        self.reflected_signal = reflected_signal

    def __str__(self):
        return f"Segment with points {self.p1} and {self.p2}"

class Line:
    def __init__(self, p1, p2, reflected_signal):
        self.p1 = p1
        self.p2 = p2
        self.reflected_signal = reflected_signal
        self.direction = util.normalize(np.subtract(self.p2, self.p1))

    def __str__(self):
        return (
            "Line with points: "
            + str(self.p1)
            + ", "
            + str(self.p2)
            + " and direction "
            + str(self.direction)
        )

# utils taken from math_util

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

# Function to parse each segment string into a tuple of points
def parse_segment(segment):
    import re
    points = re.findall(r'\[([-\d. ]+)\]', segment)
    p1 = list(map(float, points[0].split()))
    p2 = list(map(float, points[1].split()))
    return p1, p2

def visualize_circular_segments(numerical_values):
    x_vals = []
    y_vals = []
    z_vals = []

    for p1, p2 in numerical_values:
        x_vals.extend([p1[0], p2[0]])
        y_vals.extend([p1[1], p2[1]])
        z_vals.extend([p1[2], p2[2]])

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    z_vals = np.array(z_vals)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each segment
    for p1, p2 in numerical_values:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], marker='o')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Visual Representation of Circular Segments Endpoints')

    plt.show()

    
def visualize_line_segments(numerical_values):
    x_vals = []
    y_vals = []
    z_vals = []

    for p1, p2, direction in numerical_values:
        x_vals.extend([p1[0], p2[0]])
        y_vals.extend([p1[1], p2[1]])
        z_vals.extend([p1[2], p2[2]])

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    z_vals = np.array(z_vals)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each line segment
    for p1, p2, direction in numerical_values:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], marker='o')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Visual Representation of Line Segments')

    plt.show()