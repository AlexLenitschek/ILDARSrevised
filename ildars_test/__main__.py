# Collection of all tests for ildars and evaluation modules
from ildars_test import test_error_simulation
from ildars_test import inversion_test


def main():
    test_error_simulation.main()
    inversion_test.main()


if __name__ == "__main__":
    main()
