# Data class for reflection clusters
class ReflectionCluster:
    wall_normal = None
    reflected_signals = []

    def __init__(self, reflected_signals):
        self.reflected_signals = reflected_signals

    def __len__(self):
        return len(self.reflected_signals)
