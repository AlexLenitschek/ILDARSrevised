"""
Basic classes and main function of the ILDARS pipeline.
"""
from enum import Enum

from .direct_signal import DirectSignal
from .reflected_signal import ReflectedSignal
from .clustering import inversion
from . import wall_normal_vector

ClusteringAlgorithm = Enum('ClusteringAlgorithm', [
                           'INVERSION', 'GNOMONIC_PROJECTION'])
WallNormalAlgorithm = Enum('WallNormalAlgorithm', [
                           'ALL_PAIRS', 'LINEAR_ALL_PAIRS', 'DISJOINT_PAIRS', 'OVERLAPPING_PAIRS'])
WallSelectionMethod = Enum('WallSelectionMethod', [
                           'LARGEST_REFLECTION_CLUSTER', 'CLOSEST_LINES_EXTENDED'])
LocalizationAlgorithm = Enum('LocalizationAlgorithm', [
                             'MAP_TO_NORMAL_VECTOR', 'CLOSEST_LINES', 'REFLECTION_GEOMETRY', 'WALL_DIRECTION'])


def run_ildars(
        direct_signals: list[DirectSignal],
        reflected_signals: list[ReflectedSignal],
        clustering_algorithm,
        wall_normal_algorithm,
        wall_selection_algorithm,
        localization_algorithm):
    """
    Main function of the ILDARS pipeline.

    Args:
        direct_signals (list[DirectSignal]): List of all direct signals.
        reflected_signals (list[ReflectedSignal]): List of all reflected signals,
            also containing the respective time differences.

    Returns:
        The computed sender positions
    """
    # Compute reflection clusters
    reflection_clusters = compute_reflection_clusters(clustering_algorithm, reflected_signals)
    # Compute wall normal vectors. Wall normal vectors will be assigned to each reflected signal.
    for reflection_cluster in reflection_clusters:
        compute_wall_normal_vector(wall_normal_algorithm, reflection_cluster)
    # Compute and return sender positions
    return compute_sender_positions(wall_selection_algorithm, localization_algorithm, reflection_clusters, direct_signals, reflected_signals)

def compute_reflection_clusters(clustering_algorithm, reflected_signals):
    if clustering_algorithm is ClusteringAlgorithm.INVERSION:
        return inversion.compute_reflection_clusters(reflected_signals)
    else:
        raise NotImplementedError("Clustering algorithm", clustering_algorithm, "is not known or not implemented.")

def compute_wall_normal_vector(wall_normal_algorithm, reflection_cluster):
    if wall_normal_algorithm is WallNormalAlgorithm.ALL_PAIRS:
        return wall_normal_vector.compute_wall_normal_vector_all_pairs(reflection_cluster)
    elif wall_normal_algorithm is WallNormalAlgorithm.LINEAR_ALL_PAIRS:
        return wall_normal_vector.compute_wall_normal_vector_all_pairs_linear(reflection_cluster)
    elif wall_normal_algorithm is WallNormalAlgorithm.OVERLAPPING_PAIRS:
        return wall_normal_vector.compute_wall_normal_vector_overlapping_pairs(reflection_cluster)
    elif wall_normal_algorithm is WallNormalAlgorithm.DISJOINT_PAIRS:
        return wall_normal_vector.compute_wall_normal_vector_disjoint_pairs(reflection_cluster)
    else:
        raise NotImplementedError("Wall normal vector computation algorithm", wall_normal_algorithm, "is not known.")

def compute_sender_positions(wall_selection_algorithm, localization_algorithm, reflection_clusters, direct_signals, reflected_signals):
    # Closest Lines Extended algorithm does not require a separate wall selection algorithm. So we need a special case for it
    if wall_selection_algorithm is WallSelectionMethod.CLOSEST_LINES_EXTENDED:
        # TODO: implement Closest Lines Extended
        pass
    else:
        # TODO: implement "Wall" class that contains all the neccessary information
        # i.e. the wall normal vector, the reflections that were assigned to this wall
        # and also optionally contain a weighting which can be set during wall selection.
        # Then for each direct signal (i.e. each sender), we need all the walls which are
        # "available" for computing this senders position.
        pass