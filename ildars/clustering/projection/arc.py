import numpy as np

import ildars.math_utils as util


class Arc:
    def __init__(self, reflected_signal):
        v = reflected_signal.direct_signal.direction
        w = reflected_signal.direction
        delta = reflected_signal.delta
        self.start = util.normalize((delta / 2) * w)
        self.end = util.normalize(
            delta * ((w - v) / np.linalg.norm(w - v) ** 2)
        )
        self.reflected_signal = reflected_signal

    def __eq__(self, o):
        return self.reflected_signal.index == o.reflected_signal.index

    def __hash__(self):
        return hash(self.reflected_signal.index)
