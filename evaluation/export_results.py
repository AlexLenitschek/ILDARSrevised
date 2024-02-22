import sys
sys.path.append ('../ILDARSrevised')
from pathlib import Path

import numpy as np
import pandas as pd

from ildars.localization.sender_localization import STR_COMPUTED, STR_ORIGINAL

_tables = {}

STR_POS_ORIGINAL = "original position"
STR_POS_COMPUTED = "computed position"
STR_OFFSET = "offset"


# def export_experiment_results(
#     timestamp, last_iteration, algo_cl, algo_wn, algo_ws, algo_loc, positions
# ):
#     # Build up Tables if they are empty
#     if algo_cl not in _tables:
#         _tables[algo_cl] = {}
#     if algo_wn not in _tables[algo_cl]:
#         _tables[algo_cl][algo_wn] = {}
#     if algo_ws not in _tables[algo_cl][algo_wn]:
#         _tables[algo_cl][algo_wn][algo_ws] = {}
#     if algo_loc not in _tables[algo_cl][algo_wn][algo_ws]:
#         _tables[algo_cl][algo_wn][algo_ws][algo_loc] = pd.DataFrame(
#             data={STR_POS_ORIGINAL: [], STR_POS_COMPUTED: [], STR_OFFSET: []}
#         )
#     #Now append results to tables
#     positions_original = [pos[STR_ORIGINAL] for pos in positions]
#     positions_computed = [pos[STR_COMPUTED] for pos in positions]
#     #Calculate mean offset
#     positions_offsets = [
#         np.linalg.norm(pos_orig - pos_comp)
#         for pos_orig, pos_comp in zip(positions_original, positions_computed)
#     ]
#     assert (
#         len(positions_original)
#         == len(positions_computed)
#         == len(positions_offsets)
#     )
#     if not positions_offsets:  # Check if positions_offsets is empty
#         mean_offset = np.nan  # Set mean_offset to NaN if list is empty
#     else:
#         mean_offset = np.mean(positions_offsets)
#         median_offset = np.median(positions_offsets)
#     #print("Mean Offset = ", mean_offset)

#     # Now append results to tables, filtering out positions with large offsets
#     filtered_positions = []
#     for pos in positions:
#         offset = np.linalg.norm(pos[STR_ORIGINAL] - pos[STR_COMPUTED])
#         if offset <= mean_offset and offset <= median_offset:  # Include only positions with offsets below or equal to mean
#             pos_original = pos[STR_ORIGINAL]
#             pos_computed = pos[STR_COMPUTED]
#             # If offset is too large, set it to NaN
#             pos_offset = offset if offset <= mean_offset and offset <= median_offset else np.nan
#             filtered_positions.append(
#                 {
#                     STR_POS_ORIGINAL: " ".join(map(str, pos_original)),
#                     STR_POS_COMPUTED: " ".join(map(str, pos_computed)),
#                     STR_OFFSET: pos_offset
#                 }
#             )

#     new_positions = pd.DataFrame(filtered_positions)

#     _tables[algo_cl][algo_wn][algo_ws][algo_loc] = pd.concat(
#         [_tables[algo_cl][algo_wn][algo_ws][algo_loc], new_positions]
#     )

#     if last_iteration:
#         name_clustering = str(algo_cl).split(".")[-1]
#         name_wall_normal = str(algo_wn).split(".")[-1]
#         name_wall_selection = str(algo_ws).split(".")[-1]
#         name_localization = str(algo_loc).split(".")[-1]
#         res_name = (
#             f"{name_clustering}-"
#             + f"{name_wall_normal}-"
#             + f"{name_wall_selection}-"
#             + f"{name_localization}.csv"
#         )
#         result_path = "/".join(["results", timestamp, res_name])
#         filepath = Path(result_path)
#         filepath.parent.mkdir(parents=True, exist_ok=True)
#         _tables[algo_cl][algo_wn][algo_ws][algo_loc].to_csv(filepath, sep=",")
#         # Export to csv if current iteration is the last

#     return new_positions

def export_experiment_results(
    timestamp, last_iteration, algo_cl, algo_wn, algo_ws, algo_loc, positions
):
    # Build up Tables if they are empty
    if algo_cl not in _tables:
        _tables[algo_cl] = {}
    if algo_wn not in _tables[algo_cl]:
        _tables[algo_cl][algo_wn] = {}
    if algo_ws not in _tables[algo_cl][algo_wn]:
        _tables[algo_cl][algo_wn][algo_ws] = {}
    if algo_loc not in _tables[algo_cl][algo_wn][algo_ws]:
        _tables[algo_cl][algo_wn][algo_ws][algo_loc] = pd.DataFrame(
            data={STR_POS_ORIGINAL: [], STR_POS_COMPUTED: [], STR_OFFSET: []}
        )
    # Now append results to tables
    positions_original = [pos[STR_ORIGINAL] for pos in positions]
    positions_computed = [pos[STR_COMPUTED] for pos in positions]

    # Exchange this with the positions_offset if you want to clamp the results in between 0 and 100.
    # clamped_positions_offsets = [
    #     np.linalg.norm(pos_orig - pos_comp)
    #     for pos_orig, pos_comp in zip(positions_original, positions_computed)
    # ]
    # positions_offsets = np.clip(clamped_positions_offsets, a_min=0, a_max=100)
    
    # assert (
    #     len(positions_original)
    #     == len(positions_computed)
    #     == len(positions_offsets)
    # )

    positions_offsets = [
        np.linalg.norm(pos_orig - pos_comp)
        for pos_orig, pos_comp in zip(positions_original, positions_computed)
    ]
    assert (
        len(positions_original)
        == len(positions_computed)
        == len(positions_offsets)
    )

    # For exporting, transform np arrays to space separated strings
    positions_original = [
        " ".join(map(str, pos)) for pos in positions_original
    ]
    positions_computed = [
        " ".join(map(str, pos)) for pos in positions_computed
    ]
    new_positions = pd.DataFrame(
        {
            STR_POS_ORIGINAL: positions_original,
            STR_POS_COMPUTED: positions_computed,
            STR_OFFSET: positions_offsets,
        }
    )
    _tables[algo_cl][algo_wn][algo_ws][algo_loc] = pd.concat(
        [_tables[algo_cl][algo_wn][algo_ws][algo_loc], new_positions]
    )
    if last_iteration:
        name_clustering = str(algo_cl).split(".")[-1]
        name_wall_normal = str(algo_wn).split(".")[-1]
        name_wall_selection = str(algo_ws).split(".")[-1]
        name_localization = str(algo_loc).split(".")[-1]
        res_name = (
            f"{name_clustering}-"
            + f"{name_wall_normal}-"
            + f"{name_wall_selection}-"
            + f"{name_localization}.csv"
        )
        result_path = "/".join(["results", timestamp, res_name])
        filepath = Path(result_path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        _tables[algo_cl][algo_wn][algo_ws][algo_loc].to_csv(filepath, sep=",")
        # Export to csv if current iteration is the last
    return new_positions
