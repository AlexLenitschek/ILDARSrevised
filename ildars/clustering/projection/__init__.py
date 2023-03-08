# Main File of the gnomonic projection clustering algorithm
from .arc import Arc
from .hemisphere import Hemisphere


def compute_reflection_clusters(reflected_signals):
    hemispheres = Hemisphere.get_12_hemispheres()
    gnomonic_projection = compute_gnomonic_projection(
        reflected_signals, hemispheres
    )


def compute_gnomonic_projection(reflected_signals, hemispheres):
    arcs = [Arc(ref) for ref in reflected_signals]
    for arc in arcs:
        for hemi in hemispheres:
            hemi.add_arc(arc)
