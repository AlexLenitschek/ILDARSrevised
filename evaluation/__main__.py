import ildars
from . import testrooms

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

print(list(ildars.ClusteringAlgorithm))
print(testrooms.CUBE)