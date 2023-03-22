"""
Class that represents direct signals.
"""
import numpy as np


class DirectSignal:
    _direct_signal_list = []
    """
    Class for representing direct signals. For now only saves the direction.
    """

    def __init__(self, direction, original_sender_position):
        self.direction = np.array(direction)
        self.id = DirectSignal._add_direct_signal(self)
        self.computed_sender_positions = []
        self.original_sender_position = np.array(original_sender_position)

    def add_computed_position(self, computed_position):
        self.computed_sender_positions.append(computed_position)

    def __eq__(self, o):
        return self.id == o.id

    def __hash__(self):
        return hash(self.id)

    @staticmethod
    def _add_direct_signal(direct_signal):
        DirectSignal._direct_signal_list.append(direct_signal)
        return len(DirectSignal._direct_signal_list) - 1

    @staticmethod
    def get_direct_signal_by_id(id):
        return DirectSignal._direct_signal_list[id]
