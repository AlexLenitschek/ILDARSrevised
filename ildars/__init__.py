"""
Basic classes and main function of the ILDARS pipeline.
"""
from enum import Enum

from .direct_signal import DirectSignal
from .reflected_signal import ReflectedSignal
from . import clustering
from . import walls
from . import localization


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
    reflection_clusters = clustering.compute_reflection_clusters(clustering_algorithm, reflected_signals)
    # Compute wall normal vectors. Wall normal vectors will be assigned to each reflected signal.
    for reflection_cluster in reflection_clusters: 
        walls.compute_wall_normal_vector(wall_normal_algorithm, reflection_cluster)
    # Compute and return sender positions
    return localization.compute_sender_positions(wall_selection_algorithm, localization_algorithm, reflection_clusters, direct_signals, reflected_signals)
