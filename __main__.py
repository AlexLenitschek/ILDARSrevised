import sys
sys.path.append ('../ILDARSrevised')
import csv
from pathlib import Path
import datetime
import toml
import numpy as np

from ildars.localization.sender_localization import LocalizationAlgorithm
from evaluation.runner import Runner
from evaluation import testrooms
from evaluation.export_results import export_experiment_results
from evaluation.experiment_setup_parser import (
    read_algorithm_selection_from_settings,
)

# from evaluation import signal_simulation
# from evaluation import error_simulation

# Read experiment setup from settings.toml file
settings_file = open("evaluation/settings.toml", "r")
settings = toml.load(settings_file)

# Some constants for often used strings
STR_CLUSTERING = "clustering"
STR_WALL_NORMAL = "wall_normal"
STR_WALL_SELECTION = "wall_selection"
STR_LOCALIZATION = "localization"

VON_MISES_CONCENTRATION = settings["error"]["von_mises_concentration"]
DELTA_ERROR = settings["error"]["delta_error"]
WALL_ERROR = settings["error"]["wall_error"]

NUM_ITERATIONS = settings["general"]["iterations"]
NUM_SENDERS = settings["general"]["num_senders"]

# Picks which room should be used, change in the settings.toml and add new rooms here.
selected_room = settings["simulation"]["room"]
room_mapping = {
    "TEST1ROOM": testrooms.TEST1ROOM,
    "PYRAMIDROOM": testrooms.PYRAMIDROOM,
    "CONCERTHALL": testrooms.CONCERTHALL,
    "CUBE": testrooms.CUBE,
    "BIG_CUBE": testrooms.BIG_CUBE,
    "TEST1ROOMNEW": testrooms.TEST1ROOMNEW
}
room_instance = room_mapping[selected_room]

receiver_position = np.array(
    [
        settings["general"]["receiver_position"]["x"],
        settings["general"]["receiver_position"]["y"],
        settings["general"]["receiver_position"]["z"],
    ]
)

algo_sel = read_algorithm_selection_from_settings(settings)


# Generator function for selected algorithms, so we can easily iterator over
# all possible configurations
def algo_configurations(algo_sel):
    i_clustering = 0
    i_wall_normal = 0
    i_wall_selection = 0
    i_localization = 0
    while (
        i_clustering < len(algo_sel[STR_CLUSTERING])
        and i_wall_normal < len(algo_sel[STR_WALL_NORMAL])
        and i_wall_selection < len(algo_sel[STR_WALL_SELECTION])
        and i_localization < len(algo_sel[STR_LOCALIZATION])
    ):
        # We need a special case for closest lines extended
        if algo_sel[STR_LOCALIZATION][
            i_localization
        ] == LocalizationAlgorithm.CLOSEST_LINES_EXTENDED and (
            i_wall_selection < len(algo_sel[STR_WALL_SELECTION]) - 1
        ):
            # increase indices
            if i_localization < len(algo_sel[STR_LOCALIZATION]) - 1:
                i_localization += 1
            else:
                i_localization = 0
                i_wall_selection += 1
        yield {
            STR_CLUSTERING: algo_sel[STR_CLUSTERING][i_clustering],
            STR_WALL_NORMAL: algo_sel[STR_WALL_NORMAL][i_wall_normal],
            STR_WALL_SELECTION: algo_sel[STR_WALL_SELECTION][i_wall_selection],
            STR_LOCALIZATION: algo_sel[STR_LOCALIZATION][i_localization],
        }
        # increase indices
        if i_localization < len(algo_sel[STR_LOCALIZATION]) - 1:
            i_localization += 1
        elif i_wall_selection < len(algo_sel[STR_WALL_SELECTION]) - 1:
            i_localization = 0
            i_wall_selection += 1
        elif i_wall_normal < len(algo_sel[STR_WALL_NORMAL]) - 1:
            i_localization = 0
            i_wall_selection = 0
            i_wall_normal += 1
        else:
            i_localization = 0
            i_wall_selection = 0
            i_wall_normal = 0
            i_clustering += 1


def run_experiment():
    # Get the current date and time
    current_datetime = datetime.datetime.now()
    # Format the date and time as a string (e.g., "2023-10-30_14-25-30")
    timestamp = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    # Above is an Alternative to the version below as this had some 
    # syntaxerrors with Windows namingconvention
    #timestamp = str(
    #     datetime.datetime.now()#.replace(second=0, microsecond=0).isoformat()
    #)

    current_iteration = 1 
    iterations = NUM_ITERATIONS # Defined in settings.toml
    positions = []
    while current_iteration <= iterations:
        for algo_conf in algo_configurations(algo_sel):
            # print("Selected configuration:")
            # print("  Clustering algorithm:", algo_conf[STR_CLUSTERING])
            # print("  Wall normal algorithm:", algo_conf[STR_WALL_NORMAL])
            # print("  Wall selection algorithm:", algo_conf[STR_WALL_SELECTION])
            # print("  Localization algorithm:", algo_conf[STR_LOCALIZATION])
            # print("  iteration:", current_iteration)

            positions = Runner.run_experiment(
                room_instance, # Choosen in settings.toml
                receiver_position,
                NUM_SENDERS,
                VON_MISES_CONCENTRATION,
                DELTA_ERROR,
                WALL_ERROR,
                algo_conf[STR_CLUSTERING],
                algo_conf[STR_WALL_NORMAL],
                algo_conf[STR_WALL_SELECTION],
                algo_conf[STR_LOCALIZATION],
                current_iteration,
            )
            export_experiment_results(
                timestamp,
                current_iteration == iterations,
                algo_conf[STR_CLUSTERING],
                algo_conf[STR_WALL_NORMAL],
                algo_conf[STR_WALL_SELECTION],
                algo_conf[STR_LOCALIZATION],
                positions,
            )
        current_iteration += 1
    return positions


run_experiment()