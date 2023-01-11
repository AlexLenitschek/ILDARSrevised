import ildars
from . import testrooms
from . import signal_simulation
from . import error_simulation
import random

# Error setup
VON_MISES_CONCENTRATION = 1 # concentration of 1 lead so uniform distribution
DELTA_ERROR = 0.1 # 10cm standard deviation
WALL_ERROR = 0 # no wrongly assigned reflections


receiver_position = (0.1, 0.1, 0.1)
# random sender positions
sender_positions = [(random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1)) for i in range(10)]

(direct_signals, reflected_signals) = signal_simulation.generate_measurements(receiver_position,sender_positions,testrooms.CUBE)
# reflected_signals = error_simulation.simulate_reflection_error(reflected_signals, VON_MISES_CONCENTRATION, DELTA_ERROR, WALL_ERROR)

### For now: hardcoded inversion based clustering. In the future, evaulate all implemented algorithms of the ILDARS pipeline here
computed_sender_positions = ildars.run_ildars(
    direct_signals, 
    reflected_signals, 
    ildars.clustering.ClusteringAlgorithm.INVERSION, 
    ildars.wall_normal_vector.WallNormalAlgorithm.ALL_PAIRS, 
    ildars.localization.WallSelectionMethod.LARGEST_REFLECTION_CLUSTER,
    ildars.localization.LocalizationAlgorithm.WALL_DIRECTION)

print("ildars computed the following sender positions:", computed_sender_positions)
