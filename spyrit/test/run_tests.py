from test_core_meas import test_core_meas
from test_core_noise import test_core_noise
from test_core_prep import test_core_prep
from test_core_recon import test_core_recon
from test_core_warp import test_core_warp


def run_tests():
    # order matters ! Please change it if you have failing tests
    test_core_meas()
    test_core_noise()
    test_core_prep()
    test_core_warp()
    test_core_recon()  # must be after warp


if __name__ == "__main__":
    run_tests()
    print("==================================================================")
    print("              All tests passed for all core modules!              ")
    print("==================================================================")
