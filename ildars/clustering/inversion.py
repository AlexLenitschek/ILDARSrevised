# Implementation of "Lines into Bins" algorithm by Milan Müller
import numpy as np

### Hard coded thresholds
# Maximum (absolute) distance betweet two line in the same bin
LINE_TO_LINE_THRESHOLD = 0.3
# Maximum (absolution) distance from the bin's center to each of its lines
LINE_TO_BIN_THRESHOLD = 0.65
# Bins with less than (median bin size * BIN_DISCARD_THRESHOLD) lines are dropped
BIN_DISCARD_RATIO = 0.5

def invert_vector(vec):
    divisor = vec[0]**2 + vec[1]**2 + vec[2]**2
    return np.divide(vec, divisor)

def compute_cirular_segments_from_reflections(reflected_signals):
    segments = []
    for reflection in reflected_signals:
        v = reflection.direct_signal.direction
        w = reflection.direction
        delta = reflection.delta
        vw = np.subtract(w, v)
        p0 = np.divide(np.multiply(vw, delta), np.linalg.norm(vw)**2)
        p1 = np.multiply(w, delta / 2)
        segments.append([p0, p1])
    return segments

def invert_circular_segments(circular_segments):
    return [[invert_vector(vec) for vec in segment] for segment in circular_segments]

def compute_reflection_clusters(reflected_signals):
    # Compute circular segments from measurements
    circular_segments = compute_cirular_segments_from_reflections(reflected_signals)
    finite_inversion_lines = invert_circular_segments(circular_segments)
    infinite_inversion_lines = [[line[0], np.subtract(line[1], line[0])] for line in finite_inversion_lines]