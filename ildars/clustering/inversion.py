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
        self.direction = np.subtract(self.p2, self.p1)
        # normalize direction
        self.direction = np.divide(self.direction, np.linalg.norm(self.direction))
    
    def __str__(self):
        return "Line with points: " + str(self.p1) + ", " + str(self.p2) + " and direction " + str(self.direction)

class Bin:
    # default constructor, create bin with just one line (passed as an index)
    def __init__(self, line):
        self.lines = [line]
        self.center = None

    # Get the distance of a given line to the bin
    def get_distance_to_line(self, line):
        if self.center:
            line_to_center = np.subtract(self.center, line.p1)
            line_projection_length = np.dot(line_to_center, line.direction)
            # clamp projection
            line_length = np.linalg.norm(np.subtract(line.p2, line.p1))
            if line_projection_length < 0:
                line_projection_length = 0
            elif line_projection_length > line_length:
                line_projection_length = line_length
            line_projection = np.add(line.p1, np.multiply(line.direction, line_projection_length))
            direction_to_line = np.subtract(line_projection, self.center)
            current_num_lines = len(self.lines)
            shift_factor = 1 - (current_num_lines / (current_num_lines + 1))
            self.center = np.add(self.center, np.multiply(direction_to_line, shift_factor))
        # Based on first answer from https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments
        bin_line = self.lines[0]
        pA = 0 # Closest point on bin_line to line
        pB = 0 # Closest point on line to bin_line
        cross = np.cross(bin_line.direction, line.direction)
        denominator = np.linalg.norm(cross)
        if denominator < EPSILON:
            # Lines are parallel
            d0 = np.dot(bin_line.direction, np.subtract(line.p1, bin_line.p1))
            d1 = np.dot(bin_line.direction, np.subtract(line.p2, bin_line.p1))
            if d0 <= 0 >= d1:
                if np.absolute(d0) < np.absolute(d1):
                    pA = bin_line.p1
                    pB = line.p1
                else:
                    pA = bin_line.p1
                    pB = line.p2
            elif d0 >= np.linalg.norm(np.subtract(bin_line.p2, bin_line.p1)) <= d1:
                if np.absolute(d0) < np.absolute(d1):
                    pA = bin_line.p2
                    pB = line.p1
                else:
                    pA = bin_line.p2
                    pB = line.p2
            else:
                # Segments are parallel and overlapping. No unique solution exists.
                self.center = np.divide(np.add(np.add(bin_line.p1, bin_line.p2), np.add(line.p1, line.p2)), 4)
                # compute projection on lines
                bin_line_to_center = np.subtract(self.center, bin_line.p1)
                bin_line_projection_length = np.dot(bin_line_to_center, bin_line.direction)
                bin_line_projection = np.add(bin_line.p1, np.multiply(bin_line.direction, bin_line_projection_length))
                pA = bin_line_projection
                line_to_center = np.subtract(self.center, line.p1)
                line_projection_length = np.dot(line_to_center, line.direction)
                line_projection = np.add(line.p1, np.multiply(line.direction, line_projection_length))
                pB = line_projection

        else:
            # Lines are not parallel
            t = np.subtract(line.p1, bin_line.p1)

            detA = np.linalg.det([t, line.direction, cross])
            detB = np.linalg.det([t, bin_line.direction, cross])
            
            t0 = detA / denominator
            t1 = detB / denominator

            # Compute projections
            pA = np.add(bin_line.p1, np.multiply(bin_line.direction, t0))
            pB = np.add(line.p1, np.multiply(line.direction, t1))

            # Clamp projections
            if t0 < 0:
                pA = bin_line.p1
            elif t0 > np.linalg.norm(np.subtract(bin_line.p2, bin_line.p1)):
                pA = bin_line.p2
            
            if t0 < 0:
                pB = line.p1
            elif t0 > np.linalg.norm(np.subtract(line.p2, line.p1)):
                pB = line.p2

            if t0 < 0 or t0 > np.linalg.norm(np.subtract(bin_line.p2, bin_line.p1)):
                dot = np.dot(line.direction, np.subtract(pA, line.p1))
                if dot < 0:
                    dot = 0
                elif dot > np.linalg.norm(np.subtract(line.p2, line.p1)):
                    dot = np.linalg.norm(np.subtract(line.p2, line.p1))
                pB = line.p1 + np.multiply(line.direction, dot)

            if t1 < 0 or t1 > np.linalg.norm(np.subtract(line.p2, line.p1)):
                dot = np.dot(bin_line.direction, np.subtract(pB, bin_line.p1))
                if dot < 0:
                    dot = 0
                elif dot > np.linalg.norm(np.subtract(bin_line.p2, bin_line.p1)):
                    dot = np.linalg.norm(np.subtract(bin_line.p2, bin_line.p1))
                pA = bin_line.p1 + np.multiply(bin_line.direction, dot)

        # Compute center and return distance between the two points
        self.center = np.add(pA, np.divide(np.subtract(pB, pA), 2))
        return np.linalg.norm(np.subtract(pB, pA))
