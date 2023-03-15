from enum import Enum

import ildars.math_utils as util

WallSelectionMethod = Enum(
    "WallSelectionMethod",
    [
        "LARGEST_REFLECTION_CLUSTER",
        "NARROWEST_CLUSTER",
        "UNWEIGHTED_AVERAGE",
        "WEIGHTED_AVERAGE_WALL_DISTANCE",
        "CLOSEST_LINES_EXTENDED",
    ],
)


def select_walls(reflection_clusters, wall_sel_algorithm):
    if wall_sel_algorithm is WallSelectionMethod.LARGEST_REFLECTION_CLUSTER:
        return select_by_largest_cluster(reflection_clusters)
    elif wall_sel_algorithm is WallSelectionMethod.NARROWEST_CLUSTER:
        return select_by_narrowest_cluster(reflection_clusters)
    elif wall_sel_algorithm is WallSelectionMethod.UNWEIGHTED_AVERAGE:
        return reflection_clusters
    else:
        raise NotImplementedError(
            "Wall selection algorithm",
            wall_sel_algorithm,
            "is either unknown or not implemented yet.",
        )


# select the cluster with the most reflections
def select_by_largest_cluster(clusters):
    assert len(clusters) > 0
    return [max(clusters, key=len)]


# select the cluster where all reflection have the closest angular distance
# to the respective wall normal vector
def select_by_narrowest_cluster(clusters):
    return [min(clusters, key=get_cluster_distance_to_nv)]


# Get the average angular distance of a cluster to its wall normal vector
def get_cluster_distance_to_nv(cluster):
    return sum(
        [
            util.get_angular_dist(r.direction, cluster.wall_normal)
            for r in cluster.reflected_signals
        ]
    ) / len(cluster.reflected_signals)
