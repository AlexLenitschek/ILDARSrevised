"""
Class that represents a reflected signal.
Reflected signals always have a reference to a direct signal (direction) from the same
sender and the time difference between the two
"""

class ReflectedSignal:
    """
    Class for representing reflected signals. Contains the direction of the signal
    and stores a reference to the associated direct signal and the respective time
    difference between receiving the two signals.
    """
    def __init__(self, direction, direct_signal, delta, index):
        self.direction = direction
        self.direct_signal = direct_signal
        self.delta = delta
        # Index is only here for debugging purposes
        self.index = index

    def __str__(self):
        return "Reflection: #" + str(self.index)
        # + " direction: " + str(self.direction) + " direct signal: " + str(self.direct_signal)