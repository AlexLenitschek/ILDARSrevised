from ildars.clustering import ClusteringAlgorithm
from ildars.localization.wall_selection import WallSelectionMethod
from ildars.walls import WallNormalAlgorithm
from ildars.localization.sender_localization import LocalizationAlgorithm


def read_algorithm_selection_from_settings(settings):
    clustering = []
    if settings["algorithms"]["clustering"]["inversion"]:
        clustering.append(ClusteringAlgorithm.INVERSION)
    if settings["algorithms"]["clustering"]["projection"]:
        clustering.append(ClusteringAlgorithm.GNOMONIC_PROJECTION)
    wall_normal = []
    if settings["algorithms"]["wall_normal"]["all_pairs"]:
        wall_normal.append(WallNormalAlgorithm.ALL_PAIRS)
    if settings["algorithms"]["wall_normal"]["linear_all_pairs"]:
        wall_normal.append(WallNormalAlgorithm.LINEAR_ALL_PAIRS)
    if settings["algorithms"]["wall_normal"]["disjoint_pairs"]:
        wall_normal.append(WallNormalAlgorithm.DISJOINT_PAIRS)
    if settings["algorithms"]["wall_normal"]["overlapping_pairs"]:
        wall_normal.append(WallNormalAlgorithm.OVERLAPPING_PAIRS)
    wall_selection = []
    if settings["algorithms"]["wall_selection"]["largest_cluster"]:
        wall_selection.append(WallSelectionMethod.LARGEST_REFLECTION_CLUSTER)
    if settings["algorithms"]["wall_selection"]["narrowest_cluster"]:
        wall_selection.append(WallSelectionMethod.NARROWEST_CLUSTER)
    if settings["algorithms"]["wall_selection"]["unweighted_average"]:
        wall_selection.append(WallSelectionMethod.UNWEIGHTED_AVERAGE)
    if settings["algorithms"]["wall_selection"][
        "weighted_average_wall_distance"
    ]:
        wall_selection.append(
            WallSelectionMethod.WEIGHTED_AVERAGE_WALL_DISTANCE
        )
    localization = []
    if settings["algorithms"]["localization"]["wall_direction"]:
        localization.append(LocalizationAlgorithm.WALL_DIRECTION)
    if settings["algorithms"]["localization"]["map_to_wall_normal"]:
        localization.append(LocalizationAlgorithm.MAP_TO_NORMAL_VECTOR)
    if settings["algorithms"]["localization"]["reflection_geometry"]:
        localization.append(LocalizationAlgorithm.REFLECTION_GEOMETRY)
    if settings["algorithms"]["localization"]["closest_lines"]:
        localization.append(LocalizationAlgorithm.CLOSEST_LINES)
    if settings["algorithms"]["localization"]["closest_lines_extended"]:
        localization.append(LocalizationAlgorithm.CLOSEST_LINES_EXTENDED)
    return {
        "clustering": clustering,
        "wall_normal": wall_normal,
        "wall_selection": wall_selection,
        "localization": localization,
    }
