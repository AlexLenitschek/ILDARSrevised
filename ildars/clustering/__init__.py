import sys
sys.path.append ('../ILDARSrevised')
from enum import Enum
from ildars.clustering import inversion
from ildars.clustering import projection

ClusteringAlgorithm = Enum(
    "ClusteringAlgorithm", ["INVERSION", "GNOMONIC_PROJECTION"]
)


def compute_reflection_clusters(clustering_algorithm, reflected_signals):
    clusters = None
    if clustering_algorithm is ClusteringAlgorithm.INVERSION:
        clusters = inversion.compute_reflection_clusters(reflected_signals)
    elif clustering_algorithm is ClusteringAlgorithm.GNOMONIC_PROJECTION:
        clusters = projection.compute_reflection_clusters(reflected_signals)
    else:
        raise NotImplementedError(
            "Clustering algorithm",
            clustering_algorithm,
            "is not known or not implemented.",
        )
    return [c for c in clusters if len(c) > 1]
    # return list(filter(lambda c: c.size > 1, clusters))
