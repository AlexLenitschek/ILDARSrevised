### Collection of all tests for ildars and evaluation modules
import ildars
import evaluation

from . import test_error_simulation
from . import inversion_test

def main():
    test_error_simulation.main()
    inversion_test.main()

if __name__ == '__main__':
    main()