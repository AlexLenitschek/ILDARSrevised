from enum import Enum

import numpy as np

from ildars.localization.sender_localization import (
    STR_COMPUTED,
    STR_DIRECT_SIGNAL,
    STR_ORIGINAL,
)

AveragingMethod = Enum(
    "AveragingMethod", ["UNWEIGHTED", "WEIGHTED_WALL_DISTANCE"]
)

STR_POSITION = "position"
STR_WEIGHT = "weight"


def compute_average_positions_from_walls(positions_per_wall, averaging_method):
    if averaging_method is AveragingMethod.UNWEIGHTED:
        return unweighted_average(positions_per_wall)
    pass


def unweighted_average(positions_per_wall):
    # store a list with all (ids of) direct signals we add computed positions
    # to. We also want to know if we computed the average overa all computed
    # positions for a given direct signal
    direct_signals_modified = {}
    for positions in positions_per_wall:
        for position in positions:
            # keep a list of all computed positions
            direct_signal = position[STR_DIRECT_SIGNAL]
            if direct_signal not in direct_signals_modified:
                # Set averaging flag to false initially
                direct_signals_modified[direct_signal] = False
            # Now add a new computed position and its weighting to the computed
            # singal
            direct_signal.add_computed_position(
                {STR_POSITION: position[STR_COMPUTED], STR_WEIGHT: 1}
            )
            assert all(
                position[STR_ORIGINAL]
                == direct_signal.original_sender_position
            )
    return compute_average_from_list(direct_signals_modified)


def average_weighted_by_wall_dist(positions_per_wall):
    # store a list with all (ids of) direct signals we add computed positions
    # to. We also want to know if we computed the average overa all computed
    # positions for a given direct signal
    direct_signals_modified = {}
    for positions in positions_per_wall:
        for position in positions:
            # keep a list of all computed positions
            direct_signal = position[STR_DIRECT_SIGNAL]
            if direct_signal not in direct_signals_modified:
                # Set averaging flag to false initially
                direct_signals_modified[direct_signal] = False
            # Now add a new computed position and its weighting to the computed
            # singal
            direct_signal.add_computed_position(
                {STR_POSITION: position[STR_COMPUTED], STR_WEIGHT: 1}
            )
            assert all(
                position[STR_ORIGINAL]
                == direct_signal.original_sender_position
            )
    return compute_average_from_list(direct_signals_modified)


def compute_average_from_list(direct_signals_modified):
    # Go over all direct signals we added computed positions to and
    # compute the average
    results = []
    for direct_signal in direct_signals_modified:
        if not direct_signals_modified[direct_signal]:
            vector_sum = np.sum(
                np.array(
                    [
                        pos[STR_POSITION] * pos[STR_WEIGHT]
                        for pos in direct_signal.computed_sender_positions
                    ]
                ),
                axis=0,
            )
            weight_sum = np.sum(
                [
                    pos[STR_WEIGHT]
                    for pos in direct_signal.computed_sender_positions
                ]
            )
            average_position = vector_sum / weight_sum
            results.append(
                {
                    STR_COMPUTED: average_position,
                    STR_DIRECT_SIGNAL: direct_signal,
                    STR_ORIGINAL: direct_signal.original_sender_position,
                }
            )
    return results
