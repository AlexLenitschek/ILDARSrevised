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
import math
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
min_samples = 2
# Minimum cluster size, clusters smaller than min_cluster_size are discarded.
min_cluster_size = 10
cluster_size_percentage_weight = 1


# Updated order of things as the previous one had some functions mixed up.
def compute_reflection_clusters_HDB(reflected_signals):
    # Step 1: Compute the circular segments using the reflected signals (same as in DBSCAN).
    circular_segments = compute_circular_segments_from_reflections(reflected_signals)
    
    # Step 2: Compute the line segments using the circular segments. This makes clustering way easier. (same as in DBSCAN).
    line_segments = invert_circular_segments(circular_segments)
    
    # Step 3: Compute the core distances for each line segment.
    core_distances = compute_core_distances(line_segments)
    
    # Step 4: Calculate the mutual reachability distances using the core distances.
    mutual_reachability_distances = compute_mutual_reachability_distances(line_segments, core_distances)
    
    
    # Step 5: Construct the minimum spanning tree (MST) from the mutual reachability distances using kruskal.
    mst = construct_mst(mutual_reachability_distances)
    mst1 = copy.deepcopy(mst)
    mst2 = copy.deepcopy(mst)
    mst3 = copy.deepcopy(mst)
    
    ## Used for testing. Using the same MST will give us the same hierarchy.
    # Save the array to a file
    #save_array_to_file(mst, 'mst_output.txt')
    # Load the array from the file
    #saved_mst = load_array_from_file('mst_output.txt')

    # Step 6: Constructs the cluster hierarchy where every parent has the childrens ID saved.
    hierarchy = construct_hierarchy(mst1)

    # Step 7: Condense the Minimum Spanning tree into a hierarchy of cluster splits where splits with sets smaller than min_cluster_size are removed.
    # Additionally also preprocessing the single child chains to have a representative cluster of the whole chain.
    condensed_hierarchy = condense_hierarchy(hierarchy, min_cluster_size, cluster_size_percentage_weight)
    condensed_hierarchy_copy = copy.deepcopy(condensed_hierarchy)
    processed_hierarchy = preprocess_condensed(condensed_hierarchy_copy)
    processed_hierarchy_copy = copy.deepcopy(processed_hierarchy)

    # Step 8: Extracting the Clusters.
    extracted_clusters = extract_clusters(processed_hierarchy_copy)

    # Step 9: Transforming the Clusters into the desired Form. (ReflectionClusters)
    final_clusters = transform_clusters(extracted_clusters, line_segments)

    if not verify_cluster_indices(extracted_clusters, len(line_segments)):
        print("Cluster indices are inconsistent with original data")

    verify_index_consistency(mst, extracted_clusters, line_segments)

    # THE FOLLOWING IS USED TO VISUALIZE SOME OF THE STEPS ABOVE TO HAVE A BETTER UNDERSTANDING OF WHAT IS HAPPENING.
    if hdbscan_testing == True:
        # # Draws the Circular segment's connected endpoints.
        # print("Circular Segments: \n")
        # numerical_values = []
        # for circ_segment in circular_segments:
        #     print(circ_segment)
        #     p1 = circ_segment.p1
        #     p2 = circ_segment.p2
        #     numerical_values.append((p1, p2))
        # util.visualize_circular_segments(numerical_values)
        # print("############################################################################## \n")


        # # Draws the Line segment's connected endpoints.
        # print("Line Segments: \n")
        # for line in line_segments:
        #     print(line)
        #     line_numerical_values = [(line.p1, line.p2, line.direction) for line in line_segments]
        # # Visualize line segments
        # util.visualize_line_segments(line_numerical_values)
        # print("############################################################################## \n")


        # # Plots the Core Distances in a Histogram
        # print("Core Distances: \n")
        # print(core_distances)
        # plt.figure(figsize=(10, 6))
        # plt.hist(core_distances, bins=20, edgecolor='black')
        # plt.title('Histogram of Core Distances')
        # plt.xlabel('Core Distance')
        # plt.ylabel('Frequency')
        # plt.grid(True)
        # plt.show()


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
        # print("Mutual Reachability Distances Computed: \n", mutual_reachability_distances)
        # print("############################################################################## \n")
        

        # # prints the entries only that are not 0
        # util.print_non_zero_entries(mst1)
        # util.visualize_mst(mst1) # VISUALIZATION OF MST.
        # print("MST. \n", mst)
        # #duplicates = util.find_duplicates(condensed)
        # #print("Duplicates:", duplicates)

        
        # Print hierarchy clusters and calculate the total stability
        print("Hierarchy: \n")
        total_stability = 0
        for key, cluster in hierarchy.items():
            print(f"Cluster {key}: {cluster}")
            if not (math.isinf(cluster.stability) or math.isnan(cluster.stability)):
                total_stability += cluster.stability
        print()


        # Print condensed hierarchy clusters
        print("Condensed Hierarchy: \n")
        for key, cluster in condensed_hierarchy.items():
            print(f"Cluster {key}: {cluster}")
        print()


        # Print the Processed Clusters.
        print("Processed Clusters:")
        for key, cluster in processed_hierarchy.items():
            print(f"Cluster {key}: {cluster}")
        print()


        # Print the Extracted Final Clusters 
        print("Extracted Clusters:")
        for key, cluster in enumerate(extracted_clusters, 1):
            print(f"Cluster {key}: ID={cluster.cluster_id}, Elements={len(cluster.elements)}, Stability={cluster.stability}")
        print(f"\nTotal number of extracted clusters: {len(extracted_clusters)}")


        # Print the total stability (Debugging)
        print(f"Total stability: {total_stability}")
        #util.visualize_mst(mst2) # VISUALIZATION OF MST.
        # util.visualize_mst_with_clusters(mutual_reachability_distances, extracted_clusters, line_segments)
        util.visualize_mst_with_clusters(mst3, extracted_clusters, line_segments)

    return final_clusters




# Used to save a MST array. Was neccessary for testing.
def save_array_to_file(array, filename):
    np.savetxt(filename, array, fmt='%f')
    print(f"Array has been saved to {filename}")

def load_array_from_file(filename):
    saved_mst = np.loadtxt(filename)
    print(f"Array has been loaded from {filename}")
    return saved_mst


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

    edges.sort()  # Sort edges by weight

    clusters = {}
    disjoint_set = util.DisjointSet(num_segments)
    
    # Initialize single element clusters with death_distance set to 0 and birth_distance set to inf
    for i in range(num_segments):
        clusters[i] = util.ClusterNode(
            cluster_id=i,
            elements=[i],
            birth_distance=float('inf'),
            death_distance=0
        )

    cluster_counter = num_segments
    last_edge_weight = float('inf')
    for weight, u, v in edges:
        root_u = disjoint_set.find(u)
        root_v = disjoint_set.find(v)

        if root_u != root_v:
            #print(f"Merging clusters {root_u} and {root_v} with edge weight {weight}")

            # Update the birth_distance of the clusters being merged
            clusters[root_u].birth_distance = weight
            clusters[root_v].birth_distance = weight

            # Create a new cluster by merging root_u and root_v
            new_cluster = util.ClusterNode(
                cluster_id=cluster_counter,
                elements=clusters[root_u].elements + clusters[root_v].elements,
                birth_distance=float('inf'),  # Initial birth_distance for the new cluster
                death_distance=weight,  # Set death_distance to the weight of the current edge
                children=[clusters[root_u], clusters[root_v]]
            )

            # Set the parent for the child clusters
            clusters[root_u].parent = new_cluster
            clusters[root_v].parent = new_cluster

            # Union the sets in the disjoint set structure and update the new root
            new_root = disjoint_set.union(root_u, root_v)

            # Add the new cluster to the disjoint set and update parents
            disjoint_set.add_cluster(cluster_counter)
            disjoint_set.update_parent(new_cluster.elements, cluster_counter)

            # Add the new cluster to the clusters dictionary
            clusters[cluster_counter] = new_cluster

            #print(f"New cluster created with ID {cluster_counter} containing elements {new_cluster.elements} with birth_distance={new_cluster.birth_distance} and death_distance={new_cluster.death_distance}")

            # Increment the cluster counter
            cluster_counter += 1
            last_edge_weight = weight

    # Set the birth_distance and death_distance of the root cluster
    if cluster_counter > 0:
        root_cluster_id = cluster_counter - 1
        clusters[root_cluster_id].birth_distance = float('inf')
        clusters[root_cluster_id].death_distance = last_edge_weight

    # Create the final clusters dictionary sorted by cluster_id
    hierarchy_clusters = {i: clusters[i] for i in sorted(clusters.keys())}

    return hierarchy_clusters

def condense_hierarchy(hierarchy_clusters, min_cluster_size, cluster_size_percentage_weight):
    def convert_distances(cluster):
        cluster.birth_distance = 1 / cluster.birth_distance if cluster.birth_distance != float('inf') else float('inf')
        cluster.death_distance = 1 / cluster.death_distance if cluster.death_distance != 0 else float('inf')

    def calculate_stability(cluster):
        cluster.stability = abs(cluster.birth_distance - cluster.death_distance) * (len(cluster.elements) + np.log(len(cluster.elements)))# cluster_size_percentage_weight)

    def condense(cluster):
        #print(f"Condensing cluster {cluster.cluster_id} with elements {cluster.elements}")
        # Recursively condense children
        new_children = []
        for child in cluster.children:
            if child:
                condensed_child = condense(hierarchy_clusters[child.cluster_id])
                if condensed_child is not None:
                    new_children.append(condensed_child)
                #print(f"Cluster {cluster.cluster_id} new children: {[c.cluster_id for c in new_children]}")

        # Update cluster's children
        cluster.children = new_children

        # Check if this cluster meets the size threshold
        if len(cluster.elements) < min_cluster_size:
            #print(f"Cluster {cluster.cluster_id} is too small and will be removed")
            return None
        else:
            convert_distances(cluster)
            calculate_stability(cluster)
            #print(f"Updated cluster {cluster.cluster_id}: {cluster}")
            return cluster

    # Create a deepcopy of the hierarchy_clusters to avoid mutation
    hierarchy_copy = copy.deepcopy(hierarchy_clusters)

    # Find the root cluster (the one with the largest cluster_id)
    root_id = max(cluster.cluster_id for cluster in hierarchy_copy.values())
    root_cluster = hierarchy_copy[root_id]

    # Start condensing from the root
    condensed_root = condense(root_cluster)

    # Create the final condensed hierarchy
    condensed_hierarchy = {}
    def add_to_dict(cluster):
        if cluster is None:
            return
        condensed_hierarchy[cluster.cluster_id] = cluster
        for child in cluster.children:
            add_to_dict(child)

    add_to_dict(condensed_root)

    return condensed_hierarchy

# This version of preprocessing selects the cluster with the highest stability in the chain as the representative of the chain and removes all other chain clusters.
def preprocess_condensed(condensed_hierarchy):
    def find_representative_and_replace_chain(start_cluster):
        current_cluster = start_cluster
        chain = []

        # Traverse through the single child chain
        while len(current_cluster.children) == 1:
            chain.append(current_cluster)
            next_child_id = current_cluster.children[0].cluster_id
            if next_child_id not in condensed_hierarchy:
                #print(f"Error: Child cluster {next_child_id} not found in condensed_hierarchy.")
                return None
            current_cluster = condensed_hierarchy[next_child_id]

        # Append the last cluster in the chain
        chain.append(current_cluster)
        #print(f"Processing chain: {[c.cluster_id for c in chain]}")

        # # Select the first cluster in the chain with non-zero stability as the representative
        # representative = chain[0]
        # for cluster in chain:
        #     if cluster.stability != 0:
        #         representative = cluster
        #         break

        # Find the representative with the highest stability
        representative = max(chain, key=lambda cluster: cluster.stability)
        #print(f"Selected representative: {representative.cluster_id}")

        # Update the representative's children to be the last cluster's children
        representative.children = current_cluster.children    

        # Update the parent cluster to point to the representative
        parent_cluster = None
        for cluster in condensed_hierarchy.values():
            for i, child in enumerate(cluster.children):
                if child.cluster_id == start_cluster.cluster_id:
                    parent_cluster = cluster
                    cluster.children[i] = representative
                    break
            if parent_cluster:
                break      

        #if parent_cluster:
            #print(f"Parent cluster {parent_cluster.cluster_id} updated to have child {representative.cluster_id}")

        # Remove all clusters in the chain except the representative
        for cluster in chain:
            if cluster != representative and cluster.cluster_id in condensed_hierarchy:
                del condensed_hierarchy[cluster.cluster_id]
                #print(f"Removed cluster {cluster.cluster_id}")

        return representative

    # Process the entire hierarchy to replace single child chains
    for cluster_id, cluster in list(condensed_hierarchy.items()):
        if len(cluster.children) == 1:
            find_representative_and_replace_chain(cluster)

    # Identify and remove the root cluster
    root_cluster_id = max(condensed_hierarchy.keys())

    if root_cluster_id in condensed_hierarchy:
        #print(f"Removing root cluster: {root_cluster_id}")
        del condensed_hierarchy[root_cluster_id]

    return condensed_hierarchy

def extract_clusters(processed_clusters):
    final_clusters = set()
    potential_clusters = set(cluster_id for cluster_id, cluster in processed_clusters.items() if not cluster.children)

    while potential_clusters:
        cluster_id = potential_clusters.pop()
        cluster = processed_clusters[cluster_id]

        # Debug: Print the current cluster being processed
        #print(f"Processing Cluster ID={cluster_id}")

        # If cluster has no children, add to final clusters
        if not cluster.children:
            final_clusters.add(cluster)
            # Debug: Print addition of a leaf cluster
            #print(f"Added leaf Cluster ID={cluster_id} to final clusters")
            continue

        # Get parent and sibling clusters
        parent_id = None
        sibling_ids = []
        for possible_parent_id, possible_parent in processed_clusters.items():
            if cluster_id in [child.cluster_id for child in possible_parent.children]:
                parent_id = possible_parent_id
                sibling_ids = [child.cluster_id for child in possible_parent.children if child.cluster_id != cluster_id]
                break

        if not parent_id:
            final_clusters.add(cluster)
            # Debug: Print addition of an orphan cluster
            #print(f"Added orphan Cluster ID={cluster_id} to final clusters")
            continue

        parent = processed_clusters[parent_id]
        sibling_stabilities_sum = sum(processed_clusters[sibling_id].stability for sibling_id in sibling_ids)

        # Debug: Print comparison of stabilities
        #print(f"Parent ID={parent_id}, Parent Stability={parent.stability}, Siblings' Summed Stability={sibling_stabilities_sum}")

        # Compare parent stability with siblings' summed stability
        if sibling_stabilities_sum > parent.stability:
            for sibling_id in sibling_ids:
                final_clusters.add(processed_clusters[sibling_id])
                # Debug: Print addition of a sibling cluster
                #print(f"Added sibling Cluster ID={sibling_id} to final clusters")
            potential_clusters.difference_update(sibling_ids)
        else:
            potential_clusters.add(parent_id)
            potential_clusters.difference_update(sibling_ids)
            # Debug: Print re-addition of the parent cluster
            #print(f"Re-added Parent Cluster ID={parent_id} to potential clusters")

    return list(final_clusters)


# def transform_clusters(clusters, line_segments):
#     reflection_clusters = []

#     for cluster in clusters:
#         cluster_lines = [line_segments[idx] for idx in cluster.elements]  # Get the actual line segments
#         reflected_signals = lines_to_reflected_signals(cluster_lines)  # Convert lines to reflected signals
#         reflection_cluster = ReflectionCluster(reflected_signals)  # Create a ReflectionCluster object
#         reflection_clusters.append(reflection_cluster)  # Add to the list of reflection clusters

#     return reflection_clusters

def transform_clusters(clusters, line_segments):
    reflection_clusters = []
    for cluster in clusters:
        cluster_lines = []
        for idx in cluster.elements:
            if 0 <= idx < len(line_segments):
                cluster_lines.append(line_segments[idx])
            else:
                print(f"Warning: Invalid index {idx} found in cluster")
        reflected_signals = lines_to_reflected_signals(cluster_lines)
        reflection_cluster = ReflectionCluster(reflected_signals)
        reflection_clusters.append(reflection_cluster)
    return reflection_clusters

# Helper function to convert lines to reflected signals
def lines_to_reflected_signals(lines):
    return [line.reflected_signal for line in lines]

def verify_cluster_indices(clusters, total_segments):
    all_indices = set()
    for cluster in clusters:
        all_indices.update(cluster.elements)
    missing = set(range(total_segments)) - all_indices
    extra = all_indices - set(range(total_segments))
    if missing or extra:
        print(f"Warning: Inconsistent indices. Missing: {missing}, Extra: {extra}")
    return len(missing) == 0 and len(extra) == 0

def verify_index_consistency(mst, clusters, line_segments):
    mst_indices = set(range(mst.shape[0]))
    cluster_indices = set()
    for cluster in clusters:
        cluster_indices.update(cluster.elements)
    line_segment_indices = set(range(len(line_segments)))
    
    print(f"MST indices: {mst_indices}")
    print(f"Cluster indices: {cluster_indices}")
    print(f"Line segment indices: {line_segment_indices}")
    
    assert mst_indices == line_segment_indices, "MST and line segment indices mismatch"
    assert cluster_indices.issubset(line_segment_indices), "Cluster indices not a subset of line segment indices"
