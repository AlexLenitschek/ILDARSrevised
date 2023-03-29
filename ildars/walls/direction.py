import itertools

import numpy as np

import ildars.math_utils as util

STR_ALL = "all"
STR_DISJOINT = "disjoint"
STR_OVERLAPPING = "overlapping"


def compute_direction_from_pairs(
    reflection_cluster, pair_selection_method=STR_ALL
):
    reflected_signals = reflection_cluster.reflected_signals
    assert len(reflected_signals) > 1
    # Build "inner" cross porducts of each reflected signal, i.e. v x v
    inner_cross = [
        np.cross(sig.direction, sig.direct_signal.direction)
        for sig in reflected_signals
    ]
    # Initialize normal with the first two measurements
    normal = np.cross(inner_cross[0], inner_cross[1])
    inv_normal = np.multiply(normal, -1)
    # Get the "correct" direction of normal, according to v direction.
    pos_dist = util.get_angular_dist(
        normal, reflected_signals[0].direction
    ) + util.get_angular_dist(normal, reflected_signals[1].direction)
    neg_dist = util.get_angular_dist(
        inv_normal, reflected_signals[0].direction
    ) + util.get_angular_dist(inv_normal, reflected_signals[1].direction)
    # Flip normal if its inversion is closer to the wall, according to v
    # vectors from the first two measurements.
    # TODO: wouldn't this be more accurate if we average over all v vectors
    # for this comparison? It would not add asymptotical runtime (linear)
    if neg_dist < pos_dist:
        normal = inv_normal
    # Main Loop
    pairs = get_pairs(inner_cross, pair_selection_method)
    for pair in pairs:
        outer = pair[0]
        inner = pair[1]
        partial_normal = np.cross(outer, inner)
        inv_partial_normal = np.multiply(partial_normal, -1)
        if util.get_angular_dist(
            inv_partial_normal, normal
        ) < util.get_angular_dist(partial_normal, normal):
            partial_normal = inv_partial_normal
        normal += partial_normal

    normal = util.normalize(normal)
    # for reflected_signal in reflection_cluster.reflected_signals:
    #     reflected_signal.wall_normal = normal
    return normal
    # reflection_cluster.wall_normal = normal


def get_pairs(elements, method):
    if method == STR_ALL:
        return list(itertools.combinations(elements, 2))[1:]
    if method == STR_OVERLAPPING:
        return itertools.pairwise(elements[1:])
    if method == STR_DISJOINT:
        pairs = itertools.pairwise(elements[1:])
        return itertools.islice(pairs, None, None, 2)


def compute_direction_all_pairs_linear(reflected_signals):
    # TODO: implement
    # TODO: assign computed wall normal vector to each reflected signal, i.e
    # reflected_signal.wall_normal_vector = computed_wall_normal_vector
    print("Linear All Pairs not yet implemented")
    return None
