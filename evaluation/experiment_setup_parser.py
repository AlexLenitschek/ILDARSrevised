import ildars


def read_algorithm_selection_from_settings(settings):
    clustering = []
    if settings["algorithms"]["clustering"]["inversion"]:
        clustering.append(ildars.clustering.ClusteringAlgorithm.INVERSION)
    if settings["algorithms"]["clustering"]["projection"]:
        clustering.append(
            ildars.clustering.ClusteringAlgorithm.GNOMONIC_PROJECTION
        )
    wall_normal = []
    if settings["algorithms"]["wall_normal"]["all_pairs"]:
        wall_normal.append(ildars.walls.WallNormalAlgorithm.ALL_PAIRS)
    if settings["algorithms"]["wall_normal"]["linear_all_pairs"]:
        wall_normal.append(ildars.walls.WallNormalAlgorithm.LINEAR_ALL_PAIRS)
    if settings["algorithms"]["wall_normal"]["disjoint_pairs"]:
        wall_normal.append(ildars.walls.WallNormalAlgorithm.DISJOINT_PAIRS)
    if settings["algorithms"]["wall_normal"]["overlapping_pairs"]:
        wall_normal.append(ildars.walls.WallNormalAlgorithm.OVERLAPPING_PAIRS)
    wall_selection = []
    if settings["algorithms"]["wall_selection"]["largest_cluster"]:
        wall_selection.append(
            ildars.localization.WallSelectionMethod.LARGEST_REFLECTION_CLUSTER
        )
    if settings["algorithms"]["wall_selection"]["narrowest_cluster"]:
        wall_selection.append(
            ildars.localization.WallSelectionMethod.NARROWEST_CLUSTER
        )
    if settings["algorithms"]["wall_selection"]["unweighted_average"]:
        wall_selection.append(
            ildars.localization.WallSelectionMethod.UNWEIGHTED_AVERAGE
        )
    if settings["algorithms"]["wall_selection"][
        "weighted_average_wall_distance"
    ]:
        wall_selection.append(
            ildars.localization.WallSelectionMethod.WEIGHTED_AVERAGE_WALL_DISTANCE
        )
    localization = []
    if settings["algorithms"]["localization"]["wall_direction"]:
        localization.append(
            ildars.localization.LocalizationAlgorithm.WALL_DIRECTION
        )
    if settings["algorithms"]["localization"]["map_to_wall_normal"]:
        localization.append(
            ildars.localization.LocalizationAlgorithm.MAP_TO_NORMAL_VECTOR
        )
    if settings["algorithms"]["localization"]["reflection_geometry"]:
        localization.append(
            ildars.localization.LocalizationAlgorithm.REFLECTION_GEOMETRY
        )
    if settings["algorithms"]["localization"]["closest_lines"]:
        localization.append(
            ildars.localization.LocalizationAlgorithm.CLOSEST_LINES
        )
    return {
        "clustering": clustering,
        "wall_normal": wall_normal,
        "wall_selection": wall_selection,
        "localization": localization,
    }
