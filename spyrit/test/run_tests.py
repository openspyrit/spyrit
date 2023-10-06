from test_core_meas import test_core_meas
from test_core_noise import test_core_noise
from test_core_prep import test_core_prep
from test_core_recon import test_core_recon


def run_tests():
    test_core_meas()
    test_core_noise()
    test_core_prep()
    test_core_recon()


if __name__ == "__main__":
    run_tests()
