# Implementation of "Lines into Bins" algorithm by Milan Müller
import numpy as np
import math

### Only for debugging purposes
import vedo
import os

### Hard coded thresholds
# Maximum (absolute) distance betweet two line in the same bin
LINE_TO_LINE_THRESHOLD = 0.3
# Maximum (absolution) distance from the bin's center to each of its lines
LINE_TO_BIN_THRESHOLD = 0.65
# Bins with less than (median bin size * BIN_DISCARD_THRESHOLD) lines are dropped
BIN_DISCARD_RATIO = 0.5
# Very small number
EPSILON = 0.000000001

# Main function of this file. Only this function will be called from other files
def compute_reflection_clusters(reflected_signals):
    # Compute circular segments and it's inversions from measurements
    circular_segments = compute_cirular_segments_from_reflections(reflected_signals)
    lines = invert_circular_segments(circular_segments)
    
    # Initial bin
    bins = [Bin(lines[0])]
    for line in lines[1:]:
        print(line)
        current_closest_bin = None
        current_closest_dist = math.inf
        # Find closest Bin
        for bin in bins:
            pass
            # current_dist = bin.get_distance(line)
            # if current_dist < current_closest_dist:
            #     current_closest_bin = bin
            #     current_closest_dist = dist

    ### Debugging: Vizualise inversions
    room = vedo.Mesh(os.getcwd() + "/evaluation/testrooms/models/cube.obj").wireframe()
    visualization_lines = [vedo.Line(line.p1, line.p2) for line in lines]
    vedo.show(room, visualization_lines)

### Helper functions
def invert_vector(vec):
    divisor = vec[0]**2 + vec[1]**2 + vec[2]**2
    return np.divide(vec, divisor)

def compute_cirular_segments_from_reflections(reflected_signals):
    segments = []
    for reflection in reflected_signals:
        v = reflection.direct_signal.direction
        w = reflection.direction
        delta = reflection.delta
        vw = np.subtract(w, v)
        p0 = np.divide(np.multiply(vw, delta), np.linalg.norm(vw)**2)
        p1 = np.multiply(w, delta / 2)
        segments.append([p0, p1])
    return segments

def invert_circular_segments(circular_segments):
    return [Line(invert_vector(segment[0]), invert_vector(segment[1]))  for segment in circular_segments]

def is_point_on_finite_line(line, point):
    line_to_point_direction = np.subtract(point, line.p1)
    if np.dot(line.direction, line_to_point_direction) / (np.linalg.norm(line.direction) * np.linalg.norm(line_to_point_direction)) < 1 - EPSILON:
        return False
    (minX, maxX) = sorted([line.p1[0], line.p2[0]])
    (minY, maxY) = sorted([line.p1[1], line.p2[1]])
    (minZ, maxZ) = sorted([line.p1[2], line.p2[2]])
    if minX < point[0] < maxX and minY < point[1] < maxY and minZ < point[2] < maxZ:
        return True

### Classes
class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.direction = np.subtract(p2, p1)
    
    def __str__(self):
        return "Line with points: " + str(self.p1) + ", " + str(self.p2) + " and direction " + str(self.direction)

class Bin:
    lines = []
    center = None
    # default constructor, create bin with just one line (passed as an index)
    def __init__(self, line):
        self.lines.append(line)

    # Get the distance of a given line to the bin
    def get_distance(self, line):
        if self.center:
            pass
            # TODO: compute distance from (finite!) line to bin's center and return it
        # TODO: compute distance between two (finite!) lines, return it and set it as the bin center
