# tests for the error simulation. Since the error simulation uses random
# distributions, this file only prints expected, vs. actual std. deviation.
import numpy as np
from scipy.stats import vonmises_line, cramervonmises, circstd

from evaluation.error_simulation import (
    simulate_directional_error,
    random_orthogonal_vector,
)

ITERATIONS = 1000

CONCENTRATIONS = [
    {"CONCENTRATION": 3282.806, "ANGLE": 1},
    {"CONCENTRATION": 820.702, "ANGLE": 2},
    {"CONCENTRATION": 364.756, "ANGLE": 3},
    {"CONCENTRATION": 205.175, "ANGLE": 4},
    {"CONCENTRATION": 131.312, "ANGLE": 5},
    {"CONCENTRATION": 91.189, "ANGLE": 6},
    {"CONCENTRATION": 66.996, "ANGLE": 7},
    {"CONCENTRATION": 51.294, "ANGLE": 8},
    {"CONCENTRATION": 40.528, "ANGLE": 9},
    {"CONCENTRATION": 32.828, "ANGLE": 10},
]


def test_von_mises_distribution():
    for c in CONCENTRATIONS:
        angles = []
        for _ in range(ITERATIONS):
            angles.append(vonmises_line(c["CONCENTRATION"]).rvs())
        print(
            "concentration:",
            c["CONCENTRATION"],
            "expected circular standard deviation:",
            c["ANGLE"],
            "actual circular standard deviation:",
            np.rad2deg(circstd(angles)),
        )


def compute_angle(v1, v2):
    return np.arccos(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    )


def test_directional_error():
    for c in CONCENTRATIONS:
        offsets = []
        for i in range(ITERATIONS):
            # compute random vector, which is not normalized
            random_vector = np.array([np.random.random() for _ in range(3)])
            error_vector = simulate_directional_error(
                random_vector, c["CONCENTRATION"]
            )
            offsets.append(compute_angle(random_vector, error_vector))
        print(
            "concentration:",
            c["CONCENTRATION"],
            "expected circular standard deviation:",
            c["ANGLE"],
            "actual circular standard deviation:",
            np.rad2deg(circstd(offsets)),
        )


def test_random_orthogonal_vector():
    # generate one random orthogonal vector
    vec = np.array([np.random.random() for _ in range(3)])
    init_orth = random_orthogonal_vector(vec)
    angles = [
        compute_angle(
            init_orth,
            random_orthogonal_vector(vec),
        )
        for _ in range(ITERATIONS)
    ]
    print(
        "Cramer-von Mises test for",
        ITERATIONS,
        "iterations of random_orthogonal_vector",
        cramervonmises(angles, "uniform", args=(0, np.pi)),
    )


def main():
    test_random_orthogonal_vector()
    test_directional_error()
    test_von_mises_distribution()
    print("All error simulation tests run successfully")


if __name__ == "__main__":
    main()
