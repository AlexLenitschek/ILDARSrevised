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
    
    def __str__(self):
        return f"Reflection Cluster with {len(self.reflected_signals)} signals"

    # def __str__(self):
    #     lines_info = "\n".join([f"  {line}" for line in self.reflected_signals])
    #     return f"Reflection Cluster {self.id} with {len(self.reflected_signals)} signals:\n{lines_info}"

    @staticmethod
    def _get_id():
        ReflectionCluster._cluster_counter += 1
        return ReflectionCluster._cluster_counter
