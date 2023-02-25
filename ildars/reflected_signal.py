"""
Class that represents a reflected signal.
Reflected signals always have a reference to a direct signal (direction) from
the same sender and the time difference between the two
"""
import numpy as np


class ReflectedSignal:
    """
    Class for representing reflected signals. Contains the direction of the
    signal and stores a reference to the associated direct signal and the
    respective time difference between receiving the two signals.
    """

    def __init__(
        self, direction, direct_signal, delta, index, original_sender_position
    ):
        self.direction = np.array(direction)
        self.direct_signal = direct_signal
        self.delta = delta
        # Index is only here for debugging purposes
        self.index = index
        self.original_sender_position = np.array(original_sender_position)

    def __str__(self):
        return "Reflection: #" + str(self.index)
