# Implementation of DBSCAN by Alex Lenitschek
import sys
sys.path.append ('../ILDARSrevised')
import numpy as np
import math
from enum import Enum
import toml

import ildars.math_utils as util
from ildars.clustering.cluster import ReflectionCluster
from evaluation.signal_simulation import max_distance_center_to_outer_wall_center, amount_of_faces, amount_of_outer_wall_faces

# Read experiment setup from settings.toml file
settings_file = open("evaluation/settings.toml", "r")
settings = toml.load(settings_file)
# Read the selected room from the settings
number_of_senders = settings["general"]["num_senders"]
picked_room = settings["simulation"]["room"]
# Check if senderbox generation is dynamic or hardcoded - WIP
dynamic_senderbox = settings["simulation"]["dynamic_senderbox"]

# Very small number
EPSILON = 0.000000001


###############################################################################################################################################
# THIS IS FOR THE DYNAMIC ADJUSTMENT OF THE THRESHOLDS WITH THE USE OF ROOMINFORMATIONS LIKE SIZE - WORK IN PROGRESS
###############################################################################################################################################
if dynamic_senderbox == True:
    # Define the minimum and maximum room sizes and their corresponding EPS and CLUSTER_EPS values
    min_room_size = 0.85
    max_room_size = 3.50
    min_eps = 0.42
    max_eps = 0.50
    min_cluster_eps = 0.10
    max_cluster_eps = 0.50

    # Function to perform linear interpolation
    def lerp(x, x0, x1, y0, y1):
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    # Function to get EPS and CLUSTER_EPS values based on room size
    def get_values_for_room_size(room_size):
        # Clamp room size to the minimum and maximum
        room_size = min(max_room_size, max(min_room_size, room_size))
        # Perform linear interpolation for EPS and CLUSTER_EPS values
        EPS = lerp(room_size, min_room_size, max_room_size, max_eps, min_eps)
        CLUSTER_EPS = lerp(room_size, min_room_size, max_room_size, max_cluster_eps, min_cluster_eps)
        return EPS, CLUSTER_EPS

    EPS, CLUSTER_EPS = get_values_for_room_size(max_distance_center_to_outer_wall_center)
    print("EPS = ", EPS, "CLUSTER EPS = ", CLUSTER_EPS)

###############################################################################################################################################
else: # These are tested to bring the best accuracy for their respective room
    if picked_room == "PYRAMIDROOM":
        EPS = 0.50
        CLUSTER_EPS = 0.50
    elif picked_room == "CONCERTHALL":
        EPS = 0.42
        CLUSTER_EPS = 0.08
    elif picked_room == "TEST1ROOM":
        EPS = 0.42
        CLUSTER_EPS = 0.15
    else:
        EPS = 0.42
        CLUSTER_EPS = 0.15

# MINLINES consists of the average neighbor count plus this integer. This ensures that we get Clusters with more elements 
# and ignore small clusters which could be just some noises that coincidentally are next to each other.
#ADDITIONAL_NEIGHBORS = -2
# the find_core_lines() function gets repeated with MINLINES -1 until atleast this amount of core_lines is found
DESIRED_AMOUNT_OF_CORE_LINES = math.floor(number_of_senders/2) #20 for 20 senders
# Minimum amount of clusters. If there are less, then the clustering is repeated with MINLINES -1
MINIMUM_AMOUNT_OF_CLUSTERS = 2

# MINLINES = Average amount of neighbors per line divided by two and floored (minimum 2 minlines)

###########################################################################################################################################
# Step by Step Implementation of DBSCAN
#
# Preparation: 
# Define    EPS (= radius to check for neighbors) & 
#           MINLINES (=minimum amount of lines in neighborhood to make line a core line)
# PROBLEM:  - EPS might be dependent on the room size.
#               E.g. A bigger room has circular segments which are further apart 
#
# Step 1: Create line segment list with the inverted circular segments
# Step 2: Analyze the line segments in terms of their density/ closeness to each other to get MINLINES
# Step 3: Check which lines are core lines and save them in a seperate list
# Step 4: Check which lines are border lines and save them in a seperate list
# Step 5: Form the Clusters:
#           - Check for a core line if there are other core lines in the reach of it's EPS.
#           - Add these recursively for every core line that is added to the cluster until no core lines can be added anymore.
#           - These can now not be added into other clusters anymore.
#           - Now add every border line that's in the EPS range of a core line in our cluster.
#           - If there are still core lines left that are not in a cluster, repeat the process for these.
#           - Non core and non border lines are treated as noise and are ignored.
###########################################################################################################################################

# Main Function that calls all the functions to create the clusters
def compute_reflection_clusters(reflected_signals):
    circular_segments = compute_cirular_segments_from_reflections(reflected_signals)
    line_segments = invert_circular_segments(circular_segments)
    MINLINES, AVERAGE_NEIGHBORS_PER_LINE  = find_median_of_neighborcount(line_segments) # Used to get a dynamic value for MINLINES instead of an absolute
    core_lines = find_core_lines(line_segments, MINLINES, AVERAGE_NEIGHBORS_PER_LINE)
    border_lines = find_border_lines(line_segments, core_lines)
    clusters = form_clusters(core_lines, border_lines, MINLINES, AVERAGE_NEIGHBORS_PER_LINE)
    return clusters

# Step 1: Create line segment list with the inverted circular segments
# Helper Function of Milan that creates circular segments
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
    print("\nDEBUG STATISTICS FOR DBSCAN: ")
    #print("Amount of Circular Segments: ", len(segments))
    return segments

# Function that turns circular segments into line segments through inversion
def invert_circular_segments(circular_segments):
    inverted_line_segments = []
    for segment in circular_segments:
        inverted_segment = Line(
            invert_vector(segment.p1),
            invert_vector(segment.p2),
            segment.reflected_signal,
        )
        inverted_line_segments.append(inverted_segment)
    return inverted_line_segments

def find_median_of_neighborcount(line_segments):
    # Calculate the Median of the sum of neighbors for every line
    total_neighbor_sum = 0
    for line in line_segments:
        neighbor_count = 0
        for another_line in line_segments:
            a_distance = compute_distance_between_lines(line, another_line)
            if a_distance <= EPS:
                neighbor_count += 1
        total_neighbor_sum = total_neighbor_sum + neighbor_count
    
    # Calculate average number of neighbors per line
    AVERAGE_NEIGHBORS_PER_LINE = total_neighbor_sum / len(line_segments)
    # Dynamically set MINLINES based on the average number of neighbors plus some additional integer
    #MINLINES = math.floor((math.ceil(AVERAGE_NEIGHBORS_PER_LINE) + ADDITIONAL_NEIGHBORS) / 2)
    #MINLINES = math.floor(AVERAGE_NEIGHBORS_PER_LINE) + ADDITIONAL_NEIGHBORS
    MINLINES = max(math.floor(AVERAGE_NEIGHBORS_PER_LINE / 2), 2)
    print("The MINLINE value (min. amount of neighbors for line to be core_line) is: ", MINLINES)

    return MINLINES, AVERAGE_NEIGHBORS_PER_LINE

# Function that checks for all line segments if they have enough neighbors in their epsilon range EPS to be considered core_lines.
# Adaptability: If not enough core_lines are found, then this function is repeated with MINLINES - 1
def find_core_lines(line_segments, MINLINES, AVERAGE_NEIGHBORS_PER_LINE):
    core_lines = []
    total_neighbor_sum = 0
    for line in line_segments:
        neighbor_count = 0
        for other_line in line_segments:
            distance = compute_distance_between_lines(line, other_line)
            if distance <= EPS:
                neighbor_count += 1
        total_neighbor_sum += neighbor_count
        if neighbor_count >= MINLINES:
            core_lines.append(line)

    print("Average of neighbors per Line: ", total_neighbor_sum / len(line_segments))
    print("Amount of Core Lines: ", len(core_lines))

    if len(core_lines) <= DESIRED_AMOUNT_OF_CORE_LINES and MINLINES > math.ceil(AVERAGE_NEIGHBORS_PER_LINE):
        core_lines = find_core_lines(line_segments, MINLINES - 1, AVERAGE_NEIGHBORS_PER_LINE)

    # Print core lines for debugging
    # print("Core Lines:")
    # for line in core_lines:
    #     print(line)

    return core_lines

# Step 3: Identify border lines
def find_border_lines(line_segments, core_lines):
    border_lines = []
    for line in line_segments:
        if line not in core_lines:
            for core_line in core_lines:
                distance = compute_distance_between_lines(line, core_line)
                if distance <= EPS:
                    border_lines.append(line)
                    break
    print("Amount of Border lines: ", len(border_lines))
    return border_lines

# Step 4: Cluster Formation
def form_clusters(core_lines, border_lines, MINLINES, AVERAGE_NEIGHBORS_PER_LINE):
    clusters = []
    visited = set()

    def expand_cluster(cluster, line):
        # expands cluster with the element line, aslong as it isn't already part of another cluster and is in EPS range
        if line not in visited:
            visited.add(line)
            cluster.append(line)
            neighbors = [other_line for other_line in core_lines if compute_distance_between_lines(line, other_line) <= CLUSTER_EPS]
            for neighbor_line in neighbors:
                expand_cluster(cluster, neighbor_line)

    def line_in_clusters(line):
        # Check if the line is already in any cluster
        return any(line in cluster for cluster in clusters)

    for line in core_lines:
        if not line_in_clusters(line):
            new_cluster = []
            expand_cluster(new_cluster, line)
            clusters.append(new_cluster)

    for line in border_lines:
        if not line_in_clusters(line):
            for cluster in clusters:
                if any(compute_distance_between_lines(line, cluster_line) <= CLUSTER_EPS for cluster_line in cluster):
                    cluster.append(line)
                    break

    # Convert clusters of lines to clusters of ReflectionCluster
    reflection_clusters = [
        ReflectionCluster(lines_to_reflected_signals(cluster))
        for cluster in clusters
        if len(cluster) >= MINLINES
    ]

    # Check if there are enough reflection_cluster and MINLINES can be reduced
    if len(reflection_clusters) < MINIMUM_AMOUNT_OF_CLUSTERS and MINLINES > math.ceil(AVERAGE_NEIGHBORS_PER_LINE):
        print(f"Not enough Reflection clusters. Reducing MINLINES to {MINLINES - 1}")
        # Call the function again with MINLINES reduced by 1
        return form_clusters(core_lines, border_lines, MINLINES - 1, AVERAGE_NEIGHBORS_PER_LINE)
    
    # Print reflection clusters for debugging
    print("These are the Lines used in the clustering")
    print("Their sum is: ", len(core_lines) + len(border_lines))
    print("Reflection Clusters: ")
    for cluster in reflection_clusters:
        print(cluster)
    return reflection_clusters

# Helper function to convert lines to reflected signals
def lines_to_reflected_signals(lines):
    return [line.reflected_signal for line in lines]

# Helper Functions

# Compute distance between two lines for DBSCAN line clustering
def compute_distance_between_lines(line, other_line):
    # Calculate the distance between two lines based on DBSCAN requirements
    direction_distance = np.linalg.norm(np.subtract(line.direction, other_line.direction))
    if direction_distance > EPS:
        # If the direction distance is greater than epsilon, they are not considered neighbors
        return float('inf')

    # Calculate the closest points on the lines and find their distance
    closest_points = get_closest_points_on_lines(line, other_line)
    shortest_distance = np.linalg.norm(np.subtract(closest_points[0], closest_points[1]))

    return shortest_distance

# Helper functions taken from Milan Müller's Inversion Implementation
def invert_vector(vec):
    divisor = vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2
    return np.divide(vec, divisor)
    
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