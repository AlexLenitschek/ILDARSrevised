import ildars


def read_algorithm_selection_from_settings(settings):
    return {
        "clustering": [
            ildars.clustering.ClusteringAlgorithm.INVERSION
            if settings["algorithms"]["clustering"]["inversion"]
            else None,
            ildars.clustering.ClusteringAlgorithm.GNOMONIC_PROJECTION
            if settings["algorithms"]["clustering"]["projection"]
            else None,
        ],
        "wall_normal": [
            ildars.walls.WallNormalAlgorithm.ALL_PAIRS
            if settings["algorithms"]["wall_normal"]["all_pairs"]
            else None,
            ildars.walls.WallNormalAlgorithm.LINEAR_ALL_PAIRS
            if settings["algorithms"]["wall_normal"]["linear_all_pairs"]
            else None,
            ildars.walls.WallNormalAlgorithm.DISJOINT_PAIRS
            if settings["algorithms"]["wall_normal"]["disjoint_pairs"]
            else None,
            ildars.walls.WallNormalAlgorithm.OVERLAPPING_PAIRS
            if settings["algorithms"]["wall_normal"]["overlapping_pairs"]
            else None,
        ],
        "wall_selection": [
            ildars.localization.WallSelectionMethod.LARGEST_REFLECTION_CLUSTER
            if settings["algorithms"]["wall_selection"]["largest_cluster"]
            else None,
            ildars.localization.WallSelectionMethod.NARROWEST_CLUSTER
            if settings["algorithms"]["wall_selection"]["narrowest_cluster"]
            else None,
            ildars.localization.WallSelectionMethod.UNWEIGHTED_AVERAGE
            if settings["algorithms"]["wall_selection"]["unweighted_average"]
            else None,
            ildars.localization.WallSelectionMethod.WEIGHTED_AVERAGE_WALL_DISTANCE
            if settings["algorithms"]["wall_selection"][
                "weighted_average_wall_distance"
            ]
            else None,
            ildars.localization.WallSelectionMethod.CLOSEST_LINES_EXTENDED
            if settings["algorithms"]["localization"]["closest_lines_extended"]
            else None,
        ],
        "localization": [
            ildars.localization.LocalizationAlgorithm.WALL_DIRECTION
            if settings["algorithms"]["localization"]["wall_direction"]
            else None,
            ildars.localization.LocalizationAlgorithm.MAP_TO_NORMAL_VECTOR
            if settings["algorithms"]["localization"]["map_to_wall_normal"]
            else None,
            ildars.localization.LocalizationAlgorithm.REFLECTION_GEOMETRY
            if settings["algorithms"]["localization"]["reflection_geometry"]
            else None,
            ildars.localization.LocalizationAlgorithm.CLOSEST_LINES
            if settings["algorithms"]["localization"]["closest_lines"]
            else None,
        ],
    }
