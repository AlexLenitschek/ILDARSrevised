import sys
sys.path.append ('../ILDARSrevised')
from enum import Enum
from ildars.clustering import inversion
from ildars.clustering import projection
from ildars.clustering import dbscan

ClusteringAlgorithm = Enum(
    "ClusteringAlgorithm", ["INVERSION", "GNOMONIC_PROJECTION", "DBSCAN"]
)


def compute_reflection_clusters(clustering_algorithm, reflected_signals):
    clusters = None
    if clustering_algorithm is ClusteringAlgorithm.GNOMONIC_PROJECTION:
        clusters = projection.compute_reflection_clusters_GP(reflected_signals)
    elif clustering_algorithm is ClusteringAlgorithm.INVERSION:
        clusters = inversion.compute_reflection_clusters_INV(reflected_signals)
    elif clustering_algorithm is ClusteringAlgorithm.DBSCAN:
        clusters = dbscan.compute_reflection_clusters_DB(reflected_signals)    

    else:
        raise NotImplementedError(
            "Clustering algorithm",
            clustering_algorithm,
            "is not known or not implemented.",
        )
    return [c for c in clusters if len(c) > 1]
    #return list(filter(lambda c: c.size > 1, clusters))
