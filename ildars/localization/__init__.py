from enum import Enum

import ildars.localization.wall_selection as ws
import ildars.localization.sender_localization as sl
import ildars.localization.averaging as av


def compute_sender_positions(
    localization_algorithm,
    reflection_clusters,
    direct_signals,
    reflected_signals,
    wall_sel_algorithm,
):
    averaging_method = av.AveragingMethod.UNWEIGHTED
    cluster_selection = ws.select_walls(
        reflection_clusters, wall_sel_algorithm
    )
    results_per_wall = []
    for cluster in cluster_selection:
        results_per_wall.append(
            sl.compute_sender_positions_for_given_wall(
                localization_algorithm,
                cluster.wall_normal,
                cluster.reflected_signals,
            )
        )

    return av.compute_average_positions_from_walls(
        results_per_wall, averaging_method
    )
