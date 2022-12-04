import ildars
from . import testrooms
from . import signal_simulation
import random

receiver_position = (0.1, 0.1, 0.1)
# random sender positions
sender_positions = [(random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1)) for i in range(10)]

(direct_signals, reflected_signals) = signal_simulation.generate_measurements(receiver_position,sender_positions,testrooms.CUBE)

### For now: hardcoded inversion based clustering. In the future, evaulate all implemented algorithms of the ILDARS pipeline here
reflection_clusters = ildars.run_ildars(
    direct_signals, 
    reflected_signals, 
    ildars.ClusteringAlgorithm.INVERSION, 
    ildars.WallNormalAlgorithm.ALL_PAIRS, 
    ildars.WallSelectionMethod.LARGEST_REFLECTION_CLUSTER,
    ildars.LocalizationAlgorithm.CLOSEST_LINES)
