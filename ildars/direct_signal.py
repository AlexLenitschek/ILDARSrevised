"""
Class that represents direct signals.
"""


class DirectSignal:
    """
    Class for representing direct signals. For now only saves the direction.
    """
    def __init__(self, direction):
        self.direction = direction
        self.reflected_signals = []

    def add_reflected_signal(self, reflected_signal):
        self.reflected_signal.append(reflected_signal)
