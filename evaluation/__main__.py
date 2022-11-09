import ildars
from . import testrooms
from . import signal_simulation
import random

# TODO: Generate Test meaurements such as:
# (direct_signals, reflected_signals) = simulate.simulate_measurements(...)
# sender_positions = ildars.compute_sender_positions(
#     None, # direct signals
#     None, # reflected signals
#     ildars.ClusteringAlgorithm.GNOMONIC_PROJECTION,
#     ildars.WallNormalAlgorithm.ALL_PAIRS,
#     None, # wall selection method
#     ildars.LocalizationAlgorithm.CLOSEST_LINES
# )
# TODO: Evaluate computed sender positions.

receiver_position = (0.1, 0.1, 0.1)
# random sender positions
sender_positions = [(random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1)) for i in range(10)]

(direct_signals, reflected_signals) = signal_simulation.generate_measurements(receiver_position,sender_positions,testrooms.CUBE)