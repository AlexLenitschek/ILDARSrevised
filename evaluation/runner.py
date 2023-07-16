from ildars import clustering
from ildars import walls
from ildars import localization

from evaluation.signal_simulation import generate_measurements
from evaluation.error_simulation import simulate_error


class Runner:
    _simulated_signals = {}
    _clusters = {0: {}}
    _wall_normals = {0: {}}  # clusters with wall n.v.
    _actual_wall_normals = {0: {}}

    @staticmethod
    def run_experiment(
        room,
        receiver_position,
        num_senders,
        von_mises_error,
        delta_error,
        wall_error,
        algo_clustering,
        algo_wn,
        algo_wall_sel,
        algo_localization,
        iteration,
    ):
        _, reflected_signals = Runner._get_signals(
            room,
            receiver_position,
            num_senders,
            von_mises_error,
            delta_error,
            wall_error,
            iteration,
        )
        clusters = Runner._get_clusters(
            algo_clustering, iteration, reflected_signals
        )
        clusters_with_wn = [
            Runner._get_wall_normal(algo_wn, iteration, cluster)
            for cluster in clusters
        ]
        positions = localization.compute_sender_positions(
            algo_localization,
            clusters_with_wn,
            algo_wall_sel,
        )
        return positions

    @staticmethod
    def _get_signals(
        room,
        receiver_position,
        num_senders,
        von_mises_error,
        delta_error,
        wall_error,
        iteration,
    ):
        if iteration not in Runner._simulated_signals:
            del Runner._simulated_signals  # Free unused iterations from memory
            (
                direct_signals,
                reflected_signals,
                wall_nv,
            ) = generate_measurements(receiver_position, room, num_senders)
            new_measurements = simulate_error(
                direct_signals,
                reflected_signals,
                von_mises_error,
                delta_error,
                wall_error,
            )
            Runner._simulated_signals = {iteration: new_measurements}
            Runner._actual_wall_normals[iteration] = wall_nv
        return Runner._simulated_signals[iteration]

    @staticmethod
    def _get_clusters(algo, iteration, signals):
        if iteration not in Runner._clusters:
            del Runner._clusters  # Free previous iterations from memory
            Runner._clusters = {iteration: {}}
        if algo not in Runner._clusters[iteration]:
            Runner._clusters[iteration] = {
                algo: clustering.compute_reflection_clusters(algo, signals)
            }
        return Runner._clusters[iteration][algo]

    @staticmethod
    def _get_wall_normal(algo, iteration, cluster):
        if iteration not in Runner._wall_normals:
            del Runner._wall_normals  # Free unused iterations from memory
            Runner._wall_normals = {iteration: {}}
        if cluster not in Runner._wall_normals[iteration]:
            Runner._wall_normals[iteration][cluster] = {}
        if algo not in Runner._wall_normals[iteration][cluster]:
            # cluster_with_wall =
            Runner._wall_normals[iteration][cluster][
                algo
            ] = walls.compute_wall_normal_vector(algo, cluster)
        return Runner._wall_normals[iteration][cluster][algo]
