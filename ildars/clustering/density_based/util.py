import sys
sys.path.append ('../ILDARSrevised')
import numpy as np
import math
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from enum import Enum
import toml
import ildars.math_utils as util
from ildars.clustering.cluster import ReflectionCluster
import re # Regular Expressions
#from ete3 import Tree, TreeStyle, TextFace, add_face_to_node
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

# # https://pythontpoint.org/tutorial/daa/disjoint-Set-&-their-implementation-in-daa.php
# class DisjointSet:
# # A disjoint set is a data structure that keeps track of a partitioning of a set into disjoint subsets. 
# # It provides two main operations: finding the representative (root) of a set to which an element belongs using find
# # and merging two sets together using union.
#     def __init__(self, n):
#         self.parent = list(range(n)) # Initialize Parent array
#         self.rank = [0] * n # Initialize Size array with 0s 
#         self.size = n

#     def find(self, u): # Finds the representative of the set that u is an element of
#         if self.parent[u] != u: # if u is not the parent of itself then u is not the representative of its set,
#             self.parent[u] = self.find(self.parent[u]) # so we recursively call Find on its parent. 
#         return self.parent[u]

#     def union(self, u, v): # method to unite the sets containing elements u and v.
#         root_u = self.find(u) # Finds the root of the set containing u
#         root_v = self.find(v) # Finds the root of the set containing v

#         if root_u != root_v: # Checks if u and v are in different sets by comparing their roots.
#             if self.rank[root_u] > self.rank[root_v]: # If the tree rooted at root_u is taller, make root_v a child of root_u.
#                 self.parent[root_v] = root_u
#             elif self.rank[root_u] < self.rank[root_v]: # If the tree rooted at root_v is taller, make root_u a child of root_v.
#                 self.parent[root_u] = root_v
#             else: # If both trees have the same rank, make one root the parent of the other.
#                 self.parent[root_v] = root_u
#                 self.rank[root_u] += 1 # Increases the rank of the new root root_u by 1 because the height of the tree has increased.

class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))    # Initialize Parent array
        self.rank = [0] * n     # Initialize Size array with 0s

    def find(self, u):  # Finds the representative of the set that u is an element of (Basically if it is in a cluster, then this cluster will be found)
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # Path compression
        return self.parent[u]

    def union(self, u, v):  # method to unite the sets containing elements u and v.
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
                return root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
                return root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1
                return root_u
        return root_u

    def add_cluster(self, cluster_id):  # Adds a parent to a specific cluster
        self.parent.append(cluster_id)
        self.rank.append(0)

    def update_parent(self, elements, new_root):    # Updates a clusters parent
        for element in elements:
            self.parent[element] = new_root

# Class needed to construct the hierarchy.
class ClusterNode:
    def __init__(self, cluster_id, elements, birth_distance, death_distance, children=None, parent=None):
        self.cluster_id = cluster_id
        self.elements = elements
        self.birth_distance = birth_distance
        self.death_distance = death_distance
        self.children = children if children else []
        self.stability = 0
        self.parent = parent
        #self.flag = 'Default'

    def __repr__(self):
        children_ids = [child.cluster_id if child is not None else None for child in self.children]
        return (f"ClusterNode(id={self.cluster_id}, elements={self.elements}, "
                f"birth_distance={self.birth_distance}, death_distance={self.death_distance}, "
                f"children={children_ids}, stability={self.stability})")

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

# def visualize_mst(matrix):
#     G = nx.Graph()
#     for i in range(matrix.shape[0]):
#         for j in range(i + 1, matrix.shape[1]):
#             if matrix[i, j] > 0:
#                 G.add_edge(i, j, weight=matrix[i, j])
    
#     pos = nx.spring_layout(G)
#     weights = nx.get_edge_attributes(G, 'weight')
#     nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
#     plt.title("MST Visualization")
#     plt.show()

# cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'cyan', 'magenta', 'brown', 'turqoise', 'pink', 'white']  # Define more colors if needed

def visualize_mst(matrix):
    G = nx.Graph()
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            if matrix[i, j] > 0:
                G.add_edge(i, j, weight=matrix[i, j])
    
    pos = nx.kamada_kawai_layout(G, weight='weight')
    weights = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    plt.title("MST Visualization")
    plt.show()

def visualize_mst_with_clusters(mst, clusters, line_segments):
    G = nx.Graph(mst)
    node_colors = ['lightgrey'] * len(line_segments)  # Default color for noise points
    
    # Create a mapping of element to cluster index
    element_to_cluster = {}
    for i, cluster in enumerate(clusters):
        for element in cluster.elements:
            element_to_cluster[element] = i

    # Assign colors to nodes
    for node in range(len(line_segments)):
        if node in element_to_cluster:
            node_colors[node] = f"C{element_to_cluster[node]}"  # Use matplotlib color cycle for clustered points
        # Nodes not in any cluster remain 'lightgrey'

    # Use Kamada-Kawai layout for better visualization of distances
    pos = nx.kamada_kawai_layout(G, weight='weight')
    weights = nx.get_edge_attributes(G, 'weight')

    # Draw the graph with node colors
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    plt.title("MST Visualization with Clusters")
    plt.show()


def print_non_zero_entries(matrix):
    non_zero_entries = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                non_zero_entries.append((i, j, matrix[i, j]))
    print("Non-zero entries in the MST (format: (i, j, weight)):")
    for entry in non_zero_entries:
        print(entry)

# # Small testing function for condense hierarchy in HDBSCAN to see if all clusters are unique
# def find_duplicates(cluster_dict):
#     cluster_sets = {}
#     duplicates = []
    
#     for cluster_id, cluster_set in cluster_dict.items():
#         cluster_tuple = tuple(sorted(cluster_set[0]))
#         if cluster_tuple in cluster_sets:
#             duplicates.append((cluster_sets[cluster_tuple], cluster_id))
#         else:
#             cluster_sets[cluster_tuple] = cluster_id
            
#     return duplicates

def find_duplicates(cluster_dict):
    cluster_sets = {}
    duplicates = []

    for cluster_id, cluster_info in cluster_dict.items():
        if isinstance(cluster_info, dict):
            cluster_set = cluster_info['elements']
        elif isinstance(cluster_info, list):
            cluster_set = set(cluster_info)
        else:
            raise TypeError(f"Unexpected type for cluster_info: {type(cluster_info)}")

        cluster_tuple = frozenset(cluster_set)
        if cluster_tuple in cluster_sets:
            duplicates.append((cluster_sets[cluster_tuple], cluster_id))
        else:
            cluster_sets[cluster_tuple] = cluster_id

    return duplicates