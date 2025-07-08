# B_LEAST ISO 6143:2001
# Michael Wollensack METAS - 24.10.2024 - 08.07.2025

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=wildcard-import
# pylint: disable=unused-wildcard-import

from pytest import approx
from metas_b_least import *

def test_example_1():
    # See appendix B.2.1 Example 1, pages 23 - 25
    print('Example B LEAST 1\n')
    cal_data = b_read_cal_data(os.path.join(data_dir, 'b_least_1_data_cal.txt'))
    meas_data = b_read_meas_data(os.path.join(data_dir, 'b_least_1_data_meas.txt'))
    print('Linear function\n')
    b, b_cov, b_res, x, x_cov = b_test(cal_data, meas_data, b_linear_func)
    # intercept b0
    assert b[0] == approx(-3.5747e-1, abs=0.0001e-1)
    # slope b1
    assert b[1] == approx(2.4612e1, abs=0.0001e1)
    # standard uncertainty of the intercept
    assert np.sqrt(b_cov[0, 0]) == approx(1.5716e-1, abs=0.0003e-1) # 1.5713e-1
    # standard uncertainty of the slope
    assert np.sqrt(b_cov[1, 1]) == approx(4.8048e-1, abs=0.0013e-1) # 4.8035e-1
    # covariance between intercept and slope
    assert b_cov[0, 1] == approx(-5.6921e-2, abs=0.00031e-2) # -5.6890e-2
    # residual
    assert np.sum(b_res*b_res) == approx(0.6743, abs=0.0001)
    # goodness of fit
    assert np.max(np.abs(b_res)) == approx(0.568, abs=0.001)
    # mixture no 1
    assert x[0] == approx(5.9923, abs=0.0001)
    assert np.sqrt(x_cov[0, 0]) == approx(1.6377e-1, abs=0.0001e-1)
    # mixture no 2
    assert x[1] == approx(1.4409e1, abs=0.0001e1)
    assert np.sqrt(x_cov[1, 1]) == approx(3.5599e-1, abs=0.0003e-1) # 3.5597e-1
    # mixture no 3
    assert x[2] == approx(4.3943e1, abs=0.0001e1)
    assert np.sqrt(x_cov[2, 2]) == approx(1.1631, abs=0.0002) # 1.1630
    # covariance between values for mixtures 1 and 2
    assert x_cov[0, 1] == approx(1.16e-2, abs=0.01e-2)
    # covariance between values for mixtures 1 and 3
    assert x_cov[0, 2] == approx(1.48e-2, abs=0.01e-2)
    # covariance between values for mixtures 2 and 3
    assert x_cov[1, 2] == approx(1.37e-1, abs=0.01e-1)
