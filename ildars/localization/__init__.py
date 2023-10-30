import sys
sys.path.append ('../ILDARSrevised')

from enum import Enum

import ildars.localization.wall_selection as ws
import ildars.localization.sender_localization as sl
import ildars.localization.averaging as av


def compute_sender_positions(
    loc_algo,
    reflection_clusters,
    wall_sel_algorithm,
):
    averaging_method = av.AveragingMethod.UNWEIGHTED
    if (
        wall_sel_algorithm
        == ws.WallSelectionMethod.WEIGHTED_AVERAGE_WALL_DISTANCE
    ):
        averaging_method = av.AveragingMethod.WEIGHTED_WALL_DISTANCE
    # We need a special case for closest lines extended since it does not
    # operate on single walls and therefore, now averaging of wall selection
    # is required
    if loc_algo == sl.LocalizationAlgorithm.CLOSEST_LINES_EXTENDED:
        return sl.compute_sender_positions_closest_lines_extended(
            reflection_clusters
        )
    cluster_selection = ws.select_walls(
        reflection_clusters, wall_sel_algorithm
    )
    results_per_wall = []
    for cluster in cluster_selection:
        results_per_wall.append(
            sl.compute_sender_positions_for_given_wall(
                loc_algo,
                cluster.wall_normal,
                cluster.reflected_signals,
            )
        )

    return av.compute_average_positions_from_walls(
        results_per_wall, averaging_method
    )
