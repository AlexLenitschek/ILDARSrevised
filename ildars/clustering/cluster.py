# Data class for reflection clusters
class ReflectionCluster:
    _cluster_counter = 0

    wall_normal = None
    reflected_signals = []

    def __init__(self, reflected_signals):
        self.reflected_signals = reflected_signals
        self.id = ReflectionCluster._get_id()

    def __len__(self):
        return len(self.reflected_signals)

    def __eq__(self, o):
        return self.id == o.id

    def __hash__(self):
        return hash(self.id)

    @staticmethod
    def _get_id():
        ReflectionCluster._cluster_counter += 1
        return ReflectionCluster._cluster_counter
