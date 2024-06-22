import sys
sys.path.append ('../ILDARSrevised')
import copy
from enum import Enum
from ildars.clustering import inversion
from ildars.clustering import projection
from ildars.clustering import dbscan

ClusteringAlgorithm = Enum(
    "ClusteringAlgorithm", ["INVERSION", "GNOMONIC_PROJECTION", "DBSCAN"]
)


def compute_reflection_clusters(clustering_algorithm, reflected_signals):
    clusters = None
    # GP performs different when other clustering algorithms are used aswell. This is just an experiment to see if reflected_signals is changed and reused in any way
    # Deep copies to ensure each algorithm operates on its own independent data
    reflected_signalsGP = copy.deepcopy(reflected_signals)
    reflected_signalsINV = copy.deepcopy(reflected_signals)
    reflected_signalsDB = copy.deepcopy(reflected_signals)
    #reflected_signalsHDB = copy.deepcopy(reflected_signals)
    if clustering_algorithm is ClusteringAlgorithm.GNOMONIC_PROJECTION:
        clusters = projection.compute_reflection_clusters_GP(reflected_signalsGP)
    elif clustering_algorithm is ClusteringAlgorithm.INVERSION:
        clusters = inversion.compute_reflection_clusters_INV(reflected_signalsINV)
    elif clustering_algorithm is ClusteringAlgorithm.DBSCAN:
        clusters = dbscan.compute_reflection_clusters_DB(reflected_signalsDB)
    #elif clustering_algorithm is ClusteringAlgorithm.HDBSCAN:
        #clusters = dbscan.compute_reflection_clusters_HDB(reflected_signalsHDB)  

    else:
        raise NotImplementedError(
            "Clustering algorithm",
            clustering_algorithm,
            "is not known or not implemented.",
        )
    return [c for c in clusters if len(c) > 1]
    # return list(filter(lambda c: c.size > 1, clusters))
