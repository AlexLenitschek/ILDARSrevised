# Data class for reflection clusters
class ReflectionCluster:
    wall_normal = None
    reflected_signals = []

    def __init__(self, reflected_signals):
        self.reflected_signals = reflected_signals

    size = property(fget=lambda : len(reflected_signals))