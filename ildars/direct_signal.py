"""
Class that represents direct signals.
"""

class DirectSignal:
    """
    Class for representing direct signals. For now only saves the direction.
    """
    def __init__(self, direction: tuple[float, float, float], reflected_signals):
        self.direction = direction
        self.reflected_signals = reflected_signals
