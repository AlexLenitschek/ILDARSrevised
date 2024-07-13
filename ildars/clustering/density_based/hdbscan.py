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
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
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
min_samples = 15
# Minimum cluster size, clusters smaller than min_cluster_size are discarded.
min_cluster_size = 10

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

#     hierarchy = build_cluster_hierarchy(mst, n)
#     condensed_tree = condense_cluster_tree(hierarchy, min_cluster_size)
#     stable_clusters = extract_clusters(condensed_tree)
    
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

    # Step 6: Condense the Minimum Spanning tree into a hierarchy of cluster splits where splits with sets smaller than min_cluster_size are removed.
    condensed = condense_tree(mst, min_cluster_size)
    
    # Step 7: Stability calculation.

    # Step 8: Extracting the clusters with the best stability.


    # THE FOLLOWING IS USED TO VISUALIZE SOME OF THE STEPS ABOVE TO HAVE A BETTER UNDERSTANDING OF WHAT IS HAPPENING.
    if hdbscan_testing == True:
#         # print("Circular Segments: \n")
#         # numerical_values = []
#         # for circ_segment in circular_segments:
#         #     print(circ_segment)
#         #     p1 = circ_segment.p1
#         #     p2 = circ_segment.p2
#         #     numerical_values.append((p1, p2))
#         # util.visualize_circular_segments(numerical_values)
#         # print("############################################################################## \n")

#         # print("Line Segments: \n")
#         # for line in line_segments:
#         #     print(line)
#         #     line_numerical_values = [(line.p1, line.p2, line.direction) for line in line_segments]
#         # # Visualize line segments
#         # util.visualize_line_segments(line_numerical_values)
#         # print("############################################################################## \n")

#         # print("Core Distances: \n")
#         # print(core_distances)
#         # # Plotting a histogram
#         # plt.figure(figsize=(10, 6))
#         # plt.hist(core_distances, bins=20, edgecolor='black')
#         # plt.title('Histogram of Core Distances')
#         # plt.xlabel('Core Distance')
#         # plt.ylabel('Frequency')
#         # plt.grid(True)
#         # plt.show()

#         # # Plotting a line plot
#         # plt.figure(figsize=(10, 6))
#         # plt.plot(core_distances, marker='o')
#         # plt.title('Line Plot of Core Distances')
#         # plt.xlabel('Index')
#         # plt.ylabel('Core Distance')
#         # plt.grid(True)
#         # plt.show()
#         # print("############################################################################## \n")

#         # Printing the Mutual Reachability Distance Matrix
#         print("mutual_reachability_distances: \n")
#         print(mutual_reachability_distances)
#         print("############################################################################## \n")

        # Prints the mst into the terminal
        print("MST Constructed: \n", mst1)
        # prints the entries only that are not 0
        util.print_non_zero_entries(mst1)
        print("Condensed Tree:", dict(condensed))
        duplicates = util.find_duplicates(condensed)
        print("Duplicates:", duplicates)
        util.visualize_mst(mst1)

    return condensed

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

# # Step 6: Condense the mst into a hierarchy filtering out subsets with less than min_cluster_size elements
# # Implementation to condense the cluster hierarchy into clusters
def condense_tree(mst, min_cluster_size):
    num_segments = mst.shape[0]
    edges = [(mst[i, j], i, j) for i in range(num_segments) for j in range(i + 1, num_segments) if mst[i, j] > 0]
    edges.sort(reverse=True)  # Sort edges in descending order by weight

    initial_cluster = set(range(num_segments))  # The initial parent node contains all elements
    hierarchy = {0: [initial_cluster]}  # Start the hierarchy with the initial parent node
    current_clusters = {0: initial_cluster}  # Track current clusters
    cluster_id = 1  # ID for new clusters

    def get_subclusters(components):
        valid_subclusters = []
        for component in components:
            if len(component) >= min_cluster_size:
                valid_subclusters.append(component)
        return valid_subclusters

    removed_edges = set()
    existing_clusters = set()

    # Remove edges one by one and check for valid splits
    for weight, u, v in edges:
        if (u, v) in removed_edges or (v, u) in removed_edges:
            continue

        # Remove the heaviest edge
        mst[u, v] = mst[v, u] = 0
        removed_edges.add((u, v))
        removed_edges.add((v, u))

        # Convert MST to a graph and find connected components
        graph = nx.from_numpy_array(mst)
        components = list(nx.connected_components(graph))

        if len(components) <= 1:
            continue

        subclusters = get_subclusters(components)

        if len(subclusters) >= 2:
            # Find the affected cluster
            affected_cluster_id = None
            for current_cluster_id, current_cluster in current_clusters.items():
                if u in current_cluster or v in current_cluster:
                    affected_cluster_id = current_cluster_id
                    break

            if affected_cluster_id is not None:
                # Remove the affected cluster from current clusters
                del current_clusters[affected_cluster_id]

                # Add new subclusters to current clusters
                for subcluster in subclusters:
                    frozenset_subcluster = frozenset(subcluster)
                    if frozenset_subcluster not in existing_clusters:
                        current_clusters[cluster_id] = subcluster
                        hierarchy[cluster_id] = [subcluster]
                        existing_clusters.add(frozenset_subcluster)
                        cluster_id += 1

        elif len(subclusters) == 1:
            # If only one valid subcluster, update the parent node
            valid_subcluster = subclusters[0]
            remaining_nodes = initial_cluster - valid_subcluster
            if len(remaining_nodes) >= min_cluster_size:
                frozenset_valid_subcluster = frozenset(valid_subcluster)
                if frozenset_valid_subcluster not in existing_clusters:
                    hierarchy[cluster_id] = [valid_subcluster]
                    current_clusters[cluster_id] = valid_subcluster
                    existing_clusters.add(frozenset_valid_subcluster)
                    cluster_id += 1
                # Remove edges related to the invalid subcluster
                for node in remaining_nodes:
                    for i in range(num_segments):
                        if mst[node, i] > 0:
                            mst[node, i] = mst[i, node] = 0
                            removed_edges.add((node, i))
                            removed_edges.add((i, node))
        else:
            # If both resulting clusters are invalid, do not update the hierarchy and undo the edge removal
            mst[u, v] = mst[v, u] = weight
            removed_edges.remove((u, v))
            removed_edges.remove((v, u))

    return hierarchy

# # Step 7: Extract the cluster hierarchy from the MST
# # Implementation to extract the cluster hierarchy from the MST
# def extract_cluster_hierarchy(mst): 
#     # The MST obtained from the mutual reachability distances can be interpreted as a hierarchical clustering structure. 
#     # By progressively removing edges from the MST (starting with the longest), we can form a hierarchy of clusters. 
#     # The other way around would also work i.e. Adding edges and recording the clustercreations. 
#     # But I like the Idea of removing them more so we'll go with that hehe.
#     # This process effectively builds a dendrogram, where each level of the dendrogram corresponds to a different set of clusters.
#     # Extracting the cluster hierarchy is a critical step because it lays the foundation for the next step, 
#     # where stable clusters are identified and condensed. 
#     # The hierarchical structure allows HDBSCAN to select the most meaningful clusters based on stability and density, 
#     # leading to robust clustering results.
#     pass

def transform_clusters(clusters, line_segments):
    reflection_clusters = []
    
    for cluster_points in clusters.values():
        cluster_lines = [line_segments[idx] for idx in cluster_points]  # Get the actual line segments
        reflected_signals = lines_to_reflected_signals(cluster_lines)  # Convert lines to reflected signals
        reflection_cluster = ReflectionCluster(reflected_signals)  # Create a ReflectionCluster object
        reflection_clusters.append(reflection_cluster)  # Add to the list of reflection clusters
    
    return reflection_clusters

# Helper function to convert lines to reflected signals
def lines_to_reflected_signals(lines):
    return [line.reflected_signal for line in lines]