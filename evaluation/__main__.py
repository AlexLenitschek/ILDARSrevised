import toml
import csv
from pathlib import Path
import numpy as np
import datetime

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


def run_experiment(iterations=1):
    timestamp = str(
        datetime.datetime.now().replace(second=0, microsecond=0).isoformat()
    )

    current_iteration = 1
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
                testrooms.CUBE,
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
                timestamp, algo_conf, iteration, positions
            )
        current_iteration += 1
    return positions


run_experiment(NUM_ITERATIONS)


res_dir = "/".join(
    [
        "results",
        str(
            datetime.datetime.now()
            .replace(second=0, microsecond=0)
            .isoformat()
        ),
    ]
).lower()


for algo_conf in algo_configurations(algo_sel):
    positions = run_experiment(algo_conf, NUM_ITERATIONS)
    name_clustering = str(algo_conf[STR_CLUSTERING]).split(".")[-1]
    name_wall_normal = str(algo_conf[STR_WALL_NORMAL]).split(".")[-1]
    name_wall_selection = str(algo_conf[STR_WALL_SELECTION]).split(".")[-1]
    name_localization = str(algo_conf[STR_LOCALIZATION]).split(".")[-1]
    res_name = (
        f"{name_clustering}-"
        + f"{name_wall_normal}-"
        + f"{name_wall_selection}-"
        + f"{name_localization}.csv"
    )
    Path(res_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{res_dir}/{res_name}", "w", newline="") as file:
        csvwriter = csv.writer(
            file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        csvwriter.writerow(
            ["computed position", "original position", "offset"]
        )
        for position in positions:
            csvwriter.writerow(
                [
                    str(position["computed"]),
                    str(position["original"]),
                    np.linalg.norm(
                        position["computed"] - position["original"]
                    ),
                ]
            )
