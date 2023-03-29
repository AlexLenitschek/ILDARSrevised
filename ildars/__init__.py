"""
Basic classes and main function of the ILDARS pipeline.
"""
from .reflected_signal import ReflectedSignal
from . import clustering
from . import walls
from . import localization


def run_ildars(
    reflected_signals: list[ReflectedSignal],
    clustering_algorithm,
    wall_normal_algorithm,
    wall_selection_algorithm,
    localization_algorithm,
):
    """
    Main function of the ILDARS pipeline.

    Args:
        direct_signals (list[DirectSignal]): List of all direct signals.
        reflected_signals (list[ReflectedSignal]): List of all reflected
            signals, also containing the respective time differences.

    Returns:
        The computed sender positions
    """
    # Compute reflection clusters
    reflection_clusters = clustering.compute_reflection_clusters(
        clustering_algorithm, reflected_signals
    )
    # Compute wall normal vectors. Wall normal vectors will be assigned to
    # each reflected signal.
    reflection_clusters = [
        walls.compute_wall_normal_vector(wall_normal_algorithm, cluster)
        for cluster in reflection_clusters
    ]
    # Compute sender positions
    sender_positions = localization.compute_sender_positions(
        localization_algorithm,
        reflection_clusters,
        wall_selection_algorithm,
    )
    # return everything
    return (reflection_clusters, sender_positions)
