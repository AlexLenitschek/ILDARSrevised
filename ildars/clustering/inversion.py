# Implementation of "Lines into Bins" algorithm by Milan Müller
import sys
sys.path.append ('../ILDARSrevised')
import numpy as np
import math
import toml
from enum import Enum
from evaluation.signal_simulation import max_distance_center_to_outer_wall_center, amount_of_faces, amount_of_outer_wall_faces

import ildars.math_utils as util
from ildars.clustering.cluster import ReflectionCluster
# Read experiment setup from settings.toml file
settings_file = open("evaluation/settings.toml", "r")
settings = toml.load(settings_file)
# Read the selected room from the settings
picked_room = settings["simulation"]["room"]
# Check if senderbox generation is dynamic or hardcoded - WIP
dynamic_senderbox = settings["simulation"]["dynamic_senderbox"]
# Very small number
EPSILON = 0.000000001


###############################################################################################################################################
#THIS IS FOR THE DYNAMIC ADJUSTMENT OF THE THRESHOLDS WITH THE USE OF ROOMINFORMATIONS LIKE SIZE - WORK IN PROGRESS
###############################################################################################################################################
if dynamic_senderbox == True:
    # Define the minimum and maximum room sizes and their corresponding EPS and CLUSTER_EPS values
    min_room_size = 0.85
    max_room_size = 3.011
    min_LINE_TO_LINE_THRESHOLD = 0.02
    max_LINE_TO_LINE_THRESHOLD = 0.30
    min_LINE_TO_BIN_THRESHOLD = 0.08
    max_LINE_TO_BIN_THRESHOLD = 0.65
    # Function to perform linear interpolation
    def lerp(x, x0, x1, y0, y1):
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    # Function to get EPS and CLUSTER_EPS values based on room size
    def get_values_for_room_size(room_size):
        # Clamp room size to the minimum and maximum
        room_size = min(max_room_size, max(min_room_size, room_size))
        # Perform linear interpolation for EPS and CLUSTER_EPS values
        LINE_TO_LINE_THRESHOLD = lerp(room_size, min_room_size, max_room_size, max_LINE_TO_LINE_THRESHOLD, min_LINE_TO_LINE_THRESHOLD)
        LINE_TO_BIN_THRESHOLD = lerp(room_size, min_room_size, max_room_size, max_LINE_TO_BIN_THRESHOLD, min_LINE_TO_BIN_THRESHOLD)
        return LINE_TO_LINE_THRESHOLD, LINE_TO_BIN_THRESHOLD

    LINE_TO_LINE_THRESHOLD, LINE_TO_BIN_THRESHOLD = get_values_for_room_size(max_distance_center_to_outer_wall_center)
else:
    if picked_room == "PYRAMIDROOM":
        LINE_TO_LINE_THRESHOLD = 0.30
        LINE_TO_BIN_THRESHOLD = 0.65
    elif picked_room == "CONCERTHALL":
        LINE_TO_LINE_THRESHOLD = 0.04
        LINE_TO_BIN_THRESHOLD = 0.09
    elif picked_room == "TEST1ROOM":
        LINE_TO_LINE_THRESHOLD = 0.02
        LINE_TO_BIN_THRESHOLD = 0.08
    else:
        LINE_TO_LINE_THRESHOLD = 0.03
        LINE_TO_BIN_THRESHOLD = 0.09


# Hard coded thresholds
# Bins with less than median bin size * BIN_DISCARD_THRESHOLD lines are dropped
BIN_DISCARD_RATIO = 0.5



def compute_reflection_clusters(reflected_signals):
    # Compute circular segments and it's inversions from measurements
    circular_segments = compute_cirular_segments_from_reflections(
        reflected_signals
    )
    lines = invert_circular_segments(circular_segments)

    # Initial bin
    bins = [Bin(lines[0])]

    # Main Loop of Lines into Bins
    for line in lines[1:]:
        closest_single_line_bin = None
        closest_single_line_dist = Distance(
            math.inf, DistanceType.LINE_TO_LINE_DISTANCE
        )
        closest_multi_line_bin = None
        closest_multi_line_dist = Distance(
            math.inf, DistanceType.LINE_TO_BIN_DISTANCE
        )
        # Find closest Bin
        for bin in bins:
            pass
            current_dist = bin.get_distance_to_line(line)
            if (
                current_dist.type == DistanceType.LINE_TO_LINE_DISTANCE
                and current_dist < closest_single_line_dist
            ):
                closest_single_line_bin = bin
                closest_single_line_dist = current_dist
            elif (
                current_dist.type == DistanceType.LINE_TO_BIN_DISTANCE
                and current_dist < closest_multi_line_dist
            ):
                closest_multi_line_bin = bin
                closest_multi_line_dist = current_dist
        if (
            closest_single_line_bin
            and closest_single_line_dist.distance < LINE_TO_LINE_THRESHOLD
        ):
            closest_single_line_bin.add_line(line)
        elif (
            closest_multi_line_bin
            and closest_multi_line_dist.distance < LINE_TO_BIN_THRESHOLD
        ):
            closest_multi_line_bin.add_line(line)
        else:
            bins.append(Bin(line))
    # Throw away small bins
    max_bin_size = len(max([b.lines for b in bins], key=len))
    bins = [b for b in bins if len(b.lines) >= np.floor(0.25 * max_bin_size)]
    # Return found clusters.
    clusters = []
    for bin in bins:
        clusters.append(
            ReflectionCluster([line.reflected_signal for line in bin.lines])
        )
    print("\nDEBUG STATISTICS FOR INVERSION: ")
    print("Reflection Clusters:")
    for cluster in clusters:
        print(cluster)
    return clusters


# Helper functions
def invert_vector(vec):
    divisor = vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2
    return np.divide(vec, divisor)


def compute_cirular_segments_from_reflections(reflected_signals):
    segments = []
    for reflection in reflected_signals:
        v = reflection.direct_signal.direction
        w = reflection.direction
        delta = reflection.delta
        vw = np.subtract(w, v)
        p0 = np.divide(np.multiply(vw, delta), np.linalg.norm(vw) ** 2)
        p1 = np.multiply(w, delta / 2)
        segments.append(Segment(p0, p1, reflection))
    return segments


def invert_circular_segments(circular_segments):
    return [
        Line(
            invert_vector(segment.p1),
            invert_vector(segment.p2),
            segment.reflected_signal,
        )
        for segment in circular_segments
    ]


def is_point_on_finite_line(line, point):
    line_to_point_direction = np.subtract(point, line.p1)
    if (
        np.dot(line.direction, line_to_point_direction)
        / (
            np.linalg.norm(line.direction)
            * np.linalg.norm(line_to_point_direction)
        )
        < 1 - EPSILON
    ):
        return False
    (minX, maxX) = sorted([line.p1[0], line.p2[0]])
    (minY, maxY) = sorted([line.p1[1], line.p2[1]])
    (minZ, maxZ) = sorted([line.p1[2], line.p2[2]])
    if (
        minX < point[0] < maxX
        and minY < point[1] < maxY
        and minZ < point[2] < maxZ
    ):
        return True


# Classes
# Dataclass
class Segment:
    def __init__(self, p1, p2, reflected_signal):
        self.p1 = p1
        self.p2 = p2
        self.reflected_signal = reflected_signal


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


class DistanceType(Enum):
    LINE_TO_LINE_DISTANCE = 1
    LINE_TO_BIN_DISTANCE = 2


class Distance:
    def __init__(self, distance, type):
        self.distance = distance
        self.type = type

    def __lt__(self, other_distance):
        if self.type != other_distance.type:
            print("Warning, comparing distances of different types!")
        return self.distance < other_distance.distance


class Bin:
    # default constructor, create bin with just one line (passed as an index)
    def __init__(self, line):
        self.lines = [line]
        self.center = None

    # Get the distance of a given line to the bin
    def get_distance_to_line(self, line):
        if len(self.lines) > 1:
            line_projection = Bin.get_closest_point_one_line_to_point(
                line, self.center
            )
            direction_to_line = np.subtract(line_projection, self.center)
            return Distance(
                np.linalg.norm(direction_to_line),
                DistanceType.LINE_TO_BIN_DISTANCE,
            )

        (pA, pB) = Bin.get_closest_points_on_lines(self.lines[0], line)
        return Distance(
            np.linalg.norm(np.subtract(pB, pA)),
            DistanceType.LINE_TO_LINE_DISTANCE,
        )

    # Add a given line to the bin
    def add_line(self, line):
        # recompute center
        if len(self.lines) < 2:
            # bin only contained one line before, so we compute the closest
            # point between two given Lines
            (pA, pB) = Bin.get_closest_points_on_lines(self.lines[0], line)
            self.center = np.divide(np.add(pA, pB), 2)
        else:
            # adjust existing center
            num_lines_current = len(self.lines)
            adjustment_amount = 1 - (
                num_lines_current / (num_lines_current + 1)
            )
            closest_point_on_new_line = (
                Bin.get_closest_point_one_line_to_point(line, self.center)
            )
            adjustment_direction = np.subtract(
                closest_point_on_new_line, self.center
            )
            self.center += np.multiply(adjustment_direction, adjustment_amount)
        # add line
        self.lines.append(line)

    # Geometric computations
    @staticmethod
    def get_closest_point_one_line_to_point(line, point):
        line_to_center = np.subtract(point, line.p1)
        line_projection_length = np.dot(line_to_center, line.direction)
        # clamp projection
        line_length = np.linalg.norm(np.subtract(line.p2, line.p1))
        if line_projection_length < 0:
            line_projection_length = 0
        elif line_projection_length > line_length:
            line_projection_length = line_length
        return np.add(
            line.p1, np.multiply(line.direction, line_projection_length)
        )

    # Based on first answer from
    # https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments
    @staticmethod
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

    def __str__(self):
        result = (
            "Bin with "
            + str(len(self.lines))
            + " lines and center at "
            + str(self.center)
            + " lines:\n"
        )
        for i, line in enumerate(self.lines):
            result += str(i) + ": " + str(line) + "\n"
        return result
