from enum import Enum
from . import inversion
from . import projection

ClusteringAlgorithm = Enum('ClusteringAlgorithm', ['INVERSION', 'GNOMONIC_PROJECTION'])

def compute_reflection_clusters(clustering_algorithm, reflected_signals):
    if clustering_algorithm is ClusteringAlgorithm.INVERSION:
        return inversion.compute_reflection_clusters(reflected_signals)
    elif clustering_algorithm is ClusteringAlgorithm.GNOMONIC_PROJECTION:
        return projection.compute_reflection_clusters(reflected_signals)
    else:
        raise NotImplementedError("Clustering algorithm", clustering_algorithm, "is not known or not implemented.")