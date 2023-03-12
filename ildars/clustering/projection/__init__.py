# Main File of the gnomonic projection clustering algorithm
import itertools
import networkx as nx

from ildars.reflected_signal import ReflectedSignal
from ildars.clustering.cluster import ReflectionCluster
import ildars_visualization.gnomonic_projection as viz

from .arc import Arc
from .hemisphere import Hemisphere


def compute_reflection_clusters(reflected_signals):
    hemispheres = Hemisphere.get_12_hemispheres()
    compute_gnomonic_projection(reflected_signals, hemispheres)
    return find_clusters(hemispheres)


def compute_gnomonic_projection(reflected_signals, hemispheres):
    arcs = [Arc(ref) for ref in reflected_signals]
    for arc in arcs:
        for hemi in hemispheres:
            hemi.add_arc(arc)


# Find the connected components on each hemisphere
def find_clusters(hemispheres):
    intersection_graphs = [
        hemi.get_intersection_graph() for hemi in hemispheres
    ]
    # Merge all graphs pair-wise
    while len(intersection_graphs) > 1:
        intersection_graphs = [
            nx.compose(*pair)
            for pair in itertools.pairwise(intersection_graphs)
        ]
    g = intersection_graphs[0]
    # Visualization
    # viz.plot_hemisphere_connectivity_graph(g)

    # Get connected components
    components = list(nx.connected_components(g))
    # Filter out single nodes
    components = [c for c in components if len(c) > 1]
    clusters = [
        ReflectionCluster(
            [ReflectedSignal.Signals[index] for index in component]
        )
        for component in components
    ]
    return clusters
