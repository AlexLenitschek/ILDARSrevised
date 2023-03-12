import numpy as np


class Arc:
    def __init__(self, reflected_signal):
        v = reflected_signal.direct_signal.direction
        w = reflected_signal.direction
        delta = reflected_signal.delta
        start = (delta / 2) * w
        end = delta * ((w - v) / np.linalg.norm(w - v) ** 2)
        self.start = start / np.linalg.norm(start)
        self.end = end / np.linalg.norm(end)
        self.reflected_signal = reflected_signal

    def __eq__(self, o):
        return self.reflected_signal.index == o.reflected_signal.index

    def __hash__(self):
        return hash(self.reflected_signal.index)
