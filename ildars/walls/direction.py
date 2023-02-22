import numpy as np


def compute_direction_all_pairs(reflection_cluster):
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
    pos_dist = get_angular_dist(
        normal, reflected_signals[0].direction
    ) + get_angular_dist(normal, reflected_signals[1].direction)
    neg_dist = get_angular_dist(
        inv_normal, reflected_signals[0].direction
    ) + get_angular_dist(inv_normal, reflected_signals[1].direction)
    # Flip normal if its inversion is closer to the wall, according to v
    # vectors from the first two measurements.
    # TODO: wouldn't this be more accurate if we average over all v vectors
    # for this comparison? It would not add asymptotical runtime (linear)
    if neg_dist < pos_dist:
        normal = inv_normal
    # Main Loop
    for i, outer in enumerate(inner_cross):
        for j, inner in enumerate(inner_cross[i + 1 :], i + 1):
            # Skip the combination of 1st and 2nd reflection, as they were
            # used for initialization
            if i == 0 and j == 1:
                continue
            partial_normal = np.cross(outer, inner)
            inv_partial_normal = np.multiply(partial_normal, -1)
            if get_angular_dist(inv_partial_normal, normal) < get_angular_dist(
                partial_normal, normal
            ):
                partial_normal = inv_partial_normal
            # TODO: normal is normalized in each step in orignal vector. Is
            # that really neccessary?
            normal += partial_normal

    normal = np.divide(normal, np.linalg.norm(normal))
    for reflected_signal in reflection_cluster.reflected_signals:
        reflected_signal.wall_normal = normal
    reflection_cluster.wall_normal = normal


def compute_direction_all_pairs_linear(reflected_signals):
    # TODO: implement
    # TODO: assign computed wall normal vector to each reflected signal, i.e
    # reflected_signal.wall_normal_vector = computed_wall_normal_vector
    print("Linear All Pairs not yet implemented")


def compute_direction_overlapping_pairs(reflected_signals):
    # TODO: implement
    # TODO: assign computed wall normal vector to each reflected signal, i.e
    # reflected_signal.wall_normal_vector = computed_wall_normal_vector
    print("Overlapping Pairs not yet implemented")


def compute_direction_disjoint_pairs(reflected_signals):
    # TODO: implement
    # TODO: assign computed wall normal vector to each reflected signal, i.e
    # reflected_signal.wall_normal_vector = computed_wall_normal_vector
    print("Disjoint Pairs not yet implemented")


# Helper function: Get the relative angular distance between two vetors.
# 0 means the vectors are parallel, 2 means they are opposite.
def get_angular_dist(v1, v2):
    v1 = np.divide(v1, np.linalg.norm(v1))
    v2 = np.divide(v2, np.linalg.norm(v2))
    return abs(np.dot(v1, v2) - 1)
