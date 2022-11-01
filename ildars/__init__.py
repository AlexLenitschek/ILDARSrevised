"""
Basic classes and main function of the ILDARS pipeline.
"""
from enum import Enum

from .direct_signal import DirectSignal
from .reflected_signal import ReflectedSignal

ClusteringAlgorithm = Enum('ClusteringAlgorithm', ['INVERSION', 'GNOMONIC_PROJECTION'])
WallNormalAlgorithm = Enum('WallNormalAlgorithm', ['ALL_PAIRS', 'LINEAR_ALL_PAIRS', 'DISJOINT_PAIRS','OVERLAPPING_PAIRS'])
LocalizationAlgorithm = Enum('LocalizationAlgorithm', ['MAP_TO_NORMAL_VECTOR', 'CLOSEST_LINES', 'REFLECTION_GEOMETRY', 'WALL_DIRECTION'])

def compute_sender_positions(
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
    # TODO: Implement ILDARS Pipeline. Each step is indicated with one comment here
    # clusters = clustering.compute_clusters(clustering_algorithm, reflected_signals)
    # wall_normals = walls.compute_wall_normals(wall_normal_algorithm, reflected_signals)
    # wall_selection = walls.select_walls(wall_selection_algorithm, reflected_signals, clusters, wall_normals)
    # sender_positions = senders.locate_senders(localization_algorithm, wall_selection, ...) # TODO: What else do we need here?
    # return sender_positions
