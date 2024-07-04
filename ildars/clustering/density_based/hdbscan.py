# Implementation of HDBSCAN by Alex Lenitschek
import sys
sys.path.append ('../ILDARSrevised')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ildars.clustering.density_based.util as util
import toml
import plotly.graph_objects as go

import tkinter as tk
from tkinter import ttk

# Read experiment setup from settings.toml file
settings_file = open("evaluation/settings.toml", "r")
settings = toml.load(settings_file)
# Read the test settings
hdbscan_testing = settings["testing"]["hdbscan_test"]

# Threshold needed for the core distance of the line_segments. 
# core_distance for a line_segment = euclidian distance to 5th closest neighbor.
min_samples = 5


# Main Function that calls all the functions to create the clusters
def compute_reflection_clusters_HDB(reflected_signals):
    # Step 1: Compute the circular segments using the reflected signals (same as in DBSCAN)
    circular_segments = compute_cirular_segments_from_reflections(reflected_signals)
    # Step 2: Compute the line segments using the circular segments. This makes clustering way easier. (same as in DBSCAN)
    line_segments = invert_circular_segments(circular_segments)
    # Step 3: Compute the core distances for each line segment
    core_distances = compute_core_distances(line_segments)
    # Step 4: Calculate the mutual reachability distances
    mutual_reachability_distances = compute_mutual_reachability_distances(line_segments, core_distances)
    # Step 5: Construct the minimum spanning tree (MST) from the mutual reachability distances
    mst = construct_mst(mutual_reachability_distances)
    # Step 6: Extract the cluster hierarchy from the MST
    cluster_hierarchy = extract_cluster_hierarchy(mst)
    # Step 7: Condense the cluster hierarchy to form the final clusters
    clusters = condense_cluster_hierarchy(cluster_hierarchy)


    # THE FOLLOWING IS USED TO VISUALIZE SOME OF THE STEPS ABOVE TO HAVE A BETTER UNDERSTANDING OF WHAT IS HAPPENING.
    if hdbscan_testing == True:
        print("Circular Segments: \n")
        numerical_values = []
        for circ_segment in circular_segments:
            print(circ_segment)
            p1 = circ_segment.p1
            p2 = circ_segment.p2
            numerical_values.append((p1, p2))
        util.visualize_circular_segments(numerical_values)
        print("############################################################################## \n")

        print("Line Segments: \n")
        for line in line_segments:
            print(line)
            line_numerical_values = [(line.p1, line.p2, line.direction) for line in line_segments]
        # Visualize line segments
        util.visualize_line_segments(line_numerical_values)
        print("############################################################################## \n")

        print("Core Distances: \n")
        print(core_distances)
        # Plotting a histogram
        plt.figure(figsize=(10, 6))
        plt.hist(core_distances, bins=20, edgecolor='black')
        plt.title('Histogram of Core Distances')
        plt.xlabel('Core Distance')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

        # Plotting a line plot
        plt.figure(figsize=(10, 6))
        plt.plot(core_distances, marker='o')
        plt.title('Line Plot of Core Distances')
        plt.xlabel('Index')
        plt.ylabel('Core Distance')
        plt.grid(True)
        plt.show()
        print("############################################################################## \n")

        # Printing the Mutual Reachability Distance Matrix
        print("mutual_reachability_distances: \n")
        print(mutual_reachability_distances)

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
        segments.append(util.Segment(p0, p1, reflection))
    print("\nDEBUG STATISTICS FOR HDBSCAN: ")
    #print("HDBSCAN - Amount of Circular Segments: ", len(segments))
    return segments

# Step 2: Compute the line segments using the circular segments. This makes clustering way easier. (same as in DBSCAN)
# Function that turns circular segments into line segments through inversion
def invert_circular_segments(circular_segments):
    inverted_line_segments = []
    for segment in circular_segments:
        inverted_segment = util.Line(
            util.invert_vector(segment.p1),
            util.invert_vector(segment.p2),
            segment.reflected_signal,
        )
        inverted_line_segments.append(inverted_segment)
    return inverted_line_segments

# Step 3: Compute the core distances for each line segment
# Implementation to compute core distances for each line segment
def compute_core_distances(line_segments): 
    # The core distance of a point p is the distance to its k-th nearest neighbor, where k is a user-defined parameter. 
    # In HDBSCAN, this parameter is typically called min_samples. 
    # The core distance tells us how densely the point is surrounded by other points.
    # Core distances are used in the next step to compute the mutual reachability distances, 
    # which incorporate both the density around each point and the pairwise distances between points.

    core_distances = [] # List that will contain all the core distances
    for i, line in enumerate(line_segments):
        distances = [] # Initialize an empty list to store the distances between the current line segment and all other line segments.
        for j, other_line in enumerate(line_segments):
            if i != j: # Skip distance calculation for same line
                dist = util.compute_distance_between_lines(line, other_line)
                distances.append(dist)
        distances.sort() # Sorts the distance in ascending order
        if len(distances) >= min_samples:
            core_distance = distances[min_samples - 1] # distance to k-th neighbor 
        else:
            core_distance = float('inf') # No k-th neighbor means distance to k-th neighbor is infinity
        core_distances.append(core_distance)
    return core_distances

# Step 4: Calculate the mutual reachability distances
# Implementation to calculate mutual reachability distances
def compute_mutual_reachability_distances(line_segments, core_distances):
    # The mutual reachability distance between two points p and q is defined as: 
    # mutual reachability distance(p,q) = max(core_distance(p), core_distance(q), distance(p,q))
    # This distance measure incorporates the density information from the core distances and ensures that points within dense regions have smaller mutual reachability distances.
    # The mutual reachability distance combines local density information with pairwise distances to form a more robust distance measure. 
    # Used to construct the MST in the next step.

    num_segments = len(line_segments) # Initialize how big the matrix will be
    mutual_reachability_distances = np.zeros((num_segments, num_segments)) # Initialize an empty matrix to store the mutual reachability distances
    # Iterate over each pair of line segments:
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            distance = util.compute_distance_between_lines(line_segments[i], line_segments[j]) # Compute the distance between line segments i and j using distance function from util
            mrd = max(core_distances[i], core_distances[j], distance) # Compute the mutual reachability distance  
            mutual_reachability_distances[i][j] = mrd  # Store the mutual reachability distance in the matrix
            mutual_reachability_distances[j][i] = mrd  # The matrix is symmetric
    
    return mutual_reachability_distances


# Implementation to construct the MST from mutual reachability distances
def construct_mst(mutual_reachability_distances): 
    # A minimum spanning tree of a weighted graph is a subset of the edges that connects all vertices together, 
    # without any cycles, and with the minimum possible total edge weight.

    # The MST is needed to create the hierarchical structure required for the next steps in HDBSCAN. 
    # The MST represents the connectivity of points based on their mutual reachability distances, 
    # which directly influences the formation of clusters in the hierarchical clustering process.
    pass


# Implementation to extract the cluster hierarchy from the MST
def extract_cluster_hierarchy(mst): 
    # The MST obtained from the mutual reachability distances can be interpreted as a hierarchical clustering structure. 
    # By progressively removing edges from the MST (starting with the longest), we can form a hierarchy of clusters. 
    # This process effectively builds a dendrogram, where each level of the dendrogram corresponds to a different set of clusters.

    # Extracting the cluster hierarchy is a critical step because it lays the foundation for the next step, 
    # where stable clusters are identified and condensed. 
    # The hierarchical structure allows HDBSCAN to select the most meaningful clusters based on stability and density, 
    # leading to robust clustering results.
    pass


# Implementation to condense the cluster hierarchy into final clusters
def condense_cluster_hierarchy(cluster_hierarchy):
    # The goal of this step is to condense the hierarchical tree (dendrogram) created from the MST to identify clusters that are significant and stable. 
    # Stability is a measure of how long a cluster persists as we move through the hierarchy. 
    # The longer a cluster exists before it splits, the more stable it is considered.
    pass