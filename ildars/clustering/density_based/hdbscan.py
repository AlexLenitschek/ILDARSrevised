# Implementation of HDBSCAN by Alex Lenitschek
import sys
sys.path.append ('../ILDARSrevised')
import numpy as np
import matplotlib.pyplot as plt
import ildars.clustering.density_based.util as util
import toml
import logging
import pandas as pd
import networkx as nx
from collections import defaultdict 
from ildars.clustering.cluster import ReflectionCluster
import copy

# Read experiment setup from settings.toml file
settings_file = open("evaluation/settings.toml", "r")
settings = toml.load(settings_file)
# Read the test settings
hdbscan_testing = settings["testing"]["hdbscan_test"]

# Threshold needed for the core distance of the line_segments. 
# core_distance for a line_segment = euclidian distance to min_samples-th closest neighbor.
min_samples = 5
# Minimum cluster size, clusters smaller than min_cluster_size are discarded.
min_cluster_size = 5

# # Main Function that calls all the functions to create the clusters
# def compute_reflection_clusters_HDB(reflected_signals):
#     # Step 1: Compute the circular segments using the reflected signals (same as in DBSCAN)
#     circular_segments = compute_cirular_segments_from_reflections(reflected_signals)
#     # Step 2: Compute the line segments using the circular segments. This makes clustering way easier. (same as in DBSCAN)
#     line_segments = invert_circular_segments(circular_segments)
#     n = len(line_segments)
#     # Step 3: Compute the core distances for each line segment
#     core_distances = compute_core_distances(line_segments)
#     # Step 4: Calculate the mutual reachability distances
#     mutual_reachability_distances = compute_mutual_reachability_distances(line_segments, core_distances)
#     # Step 5: Construct the minimum spanning tree (MST) from the mutual reachability distances
#     mst = construct_mst(mutual_reachability_distances)
#     # # Step 6: Extract the cluster hierarchy from the MST
#     # cluster_hierarchy, cluster_stability = extract_cluster_hierarchy(mst)
#     # # Step 7: Condense the cluster hierarchy to form the final clusters
#     # clusters = condense_cluster_hierarchy(cluster_hierarchy, cluster_stability)
#     # # Extra Step: Transform the clusters into the Type used in this Pipeline (ReflectionCluster).

    
#     reflection_clusters = transform_clusters(stable_clusters, line_segments)
#     print("Clusters:", stable_clusters)  # Debug print


# Updated order of things as the previous one had some functions mixed up.
def compute_reflection_clusters_HDB(reflected_signals):
    # Step 1: Compute the circular segments using the reflected signals (same as in DBSCAN).
    circular_segments = compute_circular_segments_from_reflections(reflected_signals)
    
    # Step 2: Compute the line segments using the circular segments. This makes clustering way easier. (same as in DBSCAN).
    line_segments = invert_circular_segments(circular_segments)
    
    # Step 3: Compute the core distances for each line segment.
    core_distances = compute_core_distances(line_segments)
    
    # Step 4: Calculate the mutual reachability distances.
    mutual_reachability_distances = compute_mutual_reachability_distances(line_segments, core_distances)
    print("Mutual Reachability Distances Computed: \n", mutual_reachability_distances)
    
    # Step 5: Construct the minimum spanning tree (MST) from the mutual reachability distances.
    mst = construct_mst(mutual_reachability_distances)
    mst1 = copy.deepcopy(mst)

    # Step 6: Constructs the cluster hierarchy where every parent has the childrens ID saved.
    hierarchy = construct_hierarchy(mst1)

    # Step 7: Condense the Minimum Spanning tree into a hierarchy of cluster splits where splits with sets smaller than min_cluster_size are removed.
    # condensed_hierarchy = condense_hierarchy(hierarchy, min_cluster_size)

    # Step 8: Extracting the Clusters.
    #extracted_clusters = extract_clusters(condensed)

    # Step 9: Transforming the Clusters into the desired Form. (ReflectionClusters)
    #final_clusters = transform_clusters(stable_clusters, line_segments)

    # THE FOLLOWING IS USED TO VISUALIZE SOME OF THE STEPS ABOVE TO HAVE A BETTER UNDERSTANDING OF WHAT IS HAPPENING.
    if hdbscan_testing == True:
    #     print("Circular Segments: \n")
    #     numerical_values = []
    #     for circ_segment in circular_segments:
    #         print(circ_segment)
    #         p1 = circ_segment.p1
    #         p2 = circ_segment.p2
    #         numerical_values.append((p1, p2))
    #     util.visualize_circular_segments(numerical_values)
    #     print("############################################################################## \n")

    #     print("Line Segments: \n")
    #     for line in line_segments:
    #         print(line)
    #         line_numerical_values = [(line.p1, line.p2, line.direction) for line in line_segments]
    #     # Visualize line segments
    #     util.visualize_line_segments(line_numerical_values)
    #     print("############################################################################## \n")

    #     print("Core Distances: \n")
    #     print(core_distances)
    #     # Plotting a histogram
    #     plt.figure(figsize=(10, 6))
    #     plt.hist(core_distances, bins=20, edgecolor='black')
    #     plt.title('Histogram of Core Distances')
    #     plt.xlabel('Core Distance')
    #     plt.ylabel('Frequency')
    #     plt.grid(True)
    #     plt.show()

        # # Plotting a line plot
        # plt.figure(figsize=(10, 6))
        # plt.plot(core_distances, marker='o')
        # plt.title('Line Plot of Core Distances')
        # plt.xlabel('Index')
        # plt.ylabel('Core Distance')
        # plt.grid(True)
        # plt.show()
        # print("############################################################################## \n")

        # # Printing the Mutual Reachability Distance Matrix
        # print("mutual_reachability_distances: \n")
        # print(mutual_reachability_distances)
        # print("############################################################################## \n")

        # # Prints the mst into the terminal
        # print("MST Constructed: \n", mst1)
        # # prints the entries only that are not 0
        # util.print_non_zero_entries(mst1)

        #duplicates = util.find_duplicates(condensed)
        #print("Duplicates:", duplicates)

        util.visualize_mst(mst1)
        print("MST. \n", mst)

        # Print hierarchy clusters
        for key, cluster in hierarchy.items():
            print(f"Cluster {key}: {cluster}")


    pass
    #return hierarchy

# Step 1: Create line segment list with the inverted circular segments
# Helper Function of Milan that creates circular segments
def compute_circular_segments_from_reflections(reflected_signals):
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

    # Iterate over each unique pair of line segments:
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            distance = util.compute_distance_between_lines(line_segments[i], line_segments[j]) # Compute the distance between line segments i and j using distance function from util
            mrd = max(core_distances[i], core_distances[j], distance) # Compute the mutual reachability distance  
            mutual_reachability_distances[i][j] = mrd  # Store the mutual reachability distance in the matrix
            mutual_reachability_distances[j][i] = mrd  # The matrix is symmetric
    
    return mutual_reachability_distances

#Step 5: Construct the minimum spanning tree (MST) from the mutual reachability distances
#Implementation to construct the MST from mutual reachability distances using Kruskal's Algorithm
def construct_mst(mutual_reachability_distances):
    # A minimum spanning tree of a weighted graph is a subset of the edges that connects all vertices together, 
    # without any cycles, and with the minimum possible total edge weight.
    # The MST is needed to create the hierarchical structure required for the next steps in HDBSCAN. 
    # The MST represents the connectivity of points based on their mutual reachability distances, 
    # which directly influences the formation of clusters in the hierarchical clustering process.

    num_segments = mutual_reachability_distances.shape[0] # Determines the number of rows (also columns because symmetric).
    edges = [] # Initializes an empty list to store all the edges in the graph.

    # Convert the distance matrix to a list of edges:
    for i in range(num_segments): # Because distance from i to j is same as distance from j to i we only consider the upper triangle.
        for j in range(i + 1, num_segments):
            if mutual_reachability_distances[i, j] > 0: # Only handling the legal cases. (mutual_r_d will most likely always be >0 because core distances are usualy always positive)
                edges.append((mutual_reachability_distances[i, j], i, j)) # Adds each edge as a tuple (distance, node1, node2) to the edges list.

    edges.sort() # Sort edges by weight
    disjoint_set = util.DisjointSet(num_segments) # Initializes a Disjoint Set (Union-Find) data structure.
    mst = np.zeros((num_segments, num_segments)) # Initializes an adjacency matrix to store the MST with zeros.
    for weight, u, v in edges:
        if disjoint_set.find(u) != disjoint_set.find(v): # Checks if the current edge connects two different components (i.e., it doesn't form a cycle).
            disjoint_set.union(u, v) # Unites u and v
            mst[u, v] = mst[v, u] = weight # Adds the edge to the MST by updating the adjacency matrix.
    
    return mst


def construct_hierarchy(mst):
    num_segments = mst.shape[0]
    edges = []
    # Collect all edges from the MST matrix with positive weights (upper triangle)
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            if mst[i, j] > 0:
                edges.append((mst[i, j], i, j))

    edges.sort()    # Sort edges by weight
    min_edge_weight = edges[0][0] if edges else float('inf')    # Define smallest weight for initial birth_distance
    clusters = {}
    disjoint_set = util.DisjointSet(num_segments)
    # Initialize clusters with death_distance as min_edge_weight
    for i in range(num_segments):
        clusters[i] = util.ClusterNode(
            cluster_id=i,
            elements=[i],
            birth_distance=float('inf'),
            death_distance=min_edge_weight
        )

    cluster_counter = num_segments
    for weight, u, v in edges:
        root_u = disjoint_set.find(u)
        root_v = disjoint_set.find(v)

        if root_u != root_v:
            #print(f"Merging clusters {root_u} and {root_v} with edge weight {weight}")

            # Create a new cluster by merging root_u and root_v
            new_cluster = util.ClusterNode(
                cluster_id=cluster_counter,
                elements=clusters[root_u].elements + clusters[root_v].elements,
                birth_distance=float('inf'),
                death_distance=weight,
                children=[clusters[root_u], clusters[root_v]]
            )

            # Update the birth_distance of the merged clusters
            clusters[root_u].birth_distance = weight
            clusters[root_v].birth_distance = weight

            # Union the sets in the disjoint set structure and update the new root
            new_root = disjoint_set.union(root_u, root_v)

            # Add the new cluster to the disjoint set and update parents
            disjoint_set.add_cluster(cluster_counter)
            disjoint_set.update_parent(new_cluster.elements, cluster_counter)

            # Add the new cluster to the clusters dictionary
            clusters[cluster_counter] = new_cluster

            #print(f"New root after union: {new_root}")

            # Increment the cluster counter
            cluster_counter += 1

    # Create the final clusters dictionary sorted by cluster_id
    final_clusters = {i: clusters[i] for i in sorted(clusters.keys())}

    return final_clusters


def transform_clusters(clusters, line_segments):
    reflection_clusters = []
    
    for cluster_points in clusters:
        cluster_lines = [line_segments[idx] for idx in cluster_points]  # Get the actual line segments
        reflected_signals = lines_to_reflected_signals(cluster_lines)  # Convert lines to reflected signals
        reflection_cluster = ReflectionCluster(reflected_signals)  # Create a ReflectionCluster object
        reflection_clusters.append(reflection_cluster)  # Add to the list of reflection clusters
    
    return reflection_clusters

# Helper function to convert lines to reflected signals
def lines_to_reflected_signals(lines):
    return [line.reflected_signal for line in lines]