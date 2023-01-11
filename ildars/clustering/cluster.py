# Data class for reflection clusters
class ReflectionCluster:
    wall_normal = None
    reflected_signals = []

    def __init__(self, reflected_signals):
        self.reflected_signals = reflected_signals

    def _get_size(self):
        return len(self.reflected_signals)

    size = property(fget=_get_size)