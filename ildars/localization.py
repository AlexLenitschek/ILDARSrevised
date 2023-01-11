from enum import Enum
import operator
import numpy as np

WallSelectionMethod = Enum('WallSelectionMethod', ['LARGEST_REFLECTION_CLUSTER', 'CLOSEST_LINES_EXTENDED'])
LocalizationAlgorithm = Enum('LocalizationAlgorithm', ['MAP_TO_NORMAL_VECTOR', 'CLOSEST_LINES', 'REFLECTION_GEOMETRY', 'WALL_DIRECTION'])

def compute_sender_positions(wall_selection_algorithm, localization_algorithm, reflection_clusters, direct_signals, reflected_signals):
	if wall_selection_algorithm is WallSelectionMethod.LARGEST_REFLECTION_CLUSTER:
		return compute_sender_positions_largest_cluster(localization_algorithm, reflection_clusters, reflected_signals)
	else: 
		raise NotImplementedError("Wall selection algorithm", wall_selection_algorithm, "is either unknown or not implemented yet.")

### Wall selection methods
def compute_sender_positions_largest_cluster(localization_algorithm, reflection_clusters, reflected_signals):
	assert len(reflection_clusters) > 0
	largest_cluster = max(reflection_clusters, key=operator.attrgetter("size"))
	return compute_sender_positions_for_given_wall(localization_algorithm, largest_cluster.wall_normal, reflected_signals)

### Compute sender positions using closed formulae
def compute_sender_positions_for_given_wall(localization_algorithm, wall_normal_vector, reflected_signals):
	if localization_algorithm is LocalizationAlgorithm.WALL_DIRECTION:
		for reflected_signal in reflected_signals:
			distance = distance_wall_direction(
				np.divide(wall_normal_vector,np.linalg.norm(wall_normal_vector)), 
				reflected_signal.direct_signal.direction, 
				reflected_signal.direction, 
				reflected_signal.delta) 
			return np.multiply(reflected_signal.direct_signal.direction, distance)
			
	else:
		raise NotImplementedError("Localization algorithm", localization_algorithm, "is either unknown or not implemented yet")

### Closed formulas for computing distance of sender p
# With given normalized vector v, u, w and the delta in m
def distance_wall_direction(u, v, w, delta):
	b = np.cross(np.cross(u, v), u)
	minor = np.dot(np.subtract(v, w), b) 
	p = 0
	altP = 0
	if (minor != 0): #minor != 0
		p = np.divide(np.multiply(np.dot(w, b), np.abs(delta)), minor)
	return p

# With given normalized vectors v, w, the vector n to wall and the delta in m
def distance_map_to_normal(n, v, w, delta):
	minor = np.dot(np.add(v, w), n)
	p = 0
	if (minor != 0): #minor != 0
		upper = np.dot(np.subtract(np.multiply(2, n), np.multiply(np.abs(delta), w)), n)
		p = np.divide(upper, minor)
	return p

# With given normalized vectors u, v, w
def distance_reflection_geometry(u, n, v, w):
	b = np.cross(np.cross(u, v), u)
	minor = np.add(np.multiply(np.dot(v, n), np.dot(w, b)),
				   np.multiply(np.dot(v, b), np.dot(w, n)))
	p = 0
	if (minor != 0): #minor != 0
		upper = np.multiply(2, np.multiply(np.dot(n, n), np.dot(w, b)))
		p = np.divide(upper, minor)
	return p
