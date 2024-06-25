# Main File of the gnomonic projection clustering algorithm
import sys
sys.path.append ('../ILDARSrevised')

from ildars.reflected_signal import ReflectedSignal
import ildars.clustering.density_based.dbscan as dbscan
#import ildars.clustering.density_based.hdbscan as hdbscan

def compute_reflection_clusters_DB(reflected_signals):
    clusters = dbscan.compute_reflection_clusters_in_DB(reflected_signals)
    ReflectedSignal.clear_signals
    return clusters

# def compute_reflection_clusters_HDB(reflected_signals):
#     clusters = hdbscan.compute_reflection_clusters_in_HDB(reflected_signals)
#     return clusters
