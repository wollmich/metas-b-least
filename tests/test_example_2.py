# B_LEAST ISO 6143:2001
# Michael Wollensack METAS - 24.10.2024 - 02.12.2024

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=wildcard-import
# pylint: disable=unused-wildcard-import

from pytest import approx
from metas_b_least import *

def test_example_2_linear_func():
    # See appendix B.2.2 Example 2, pages 26 - 28
    print('Example B LEAST 2\n')
    cal_data = b_read_cal_data(os.path.join(data_dir, 'b_least_2_data_cal.txt'))
    meas_data = b_read_meas_data(os.path.join(data_dir, 'b_least_2_data_meas.txt'))
    print('Linear function\n')
    b, b_cov, b_res, x, x_cov = b_test(cal_data, meas_data, b_linear_func)
    # intercept b0
    assert b[0] == approx(3.9189e-4, abs=0.0622e-4) # 3.9810e-4
    # slope b1
    assert b[1] == approx(2.4286e-5, abs=0.0001e-5) # 2.4285e-4
    # standard uncertainty of the intercept
    assert np.sqrt(b_cov[0, 0]) == approx(1.1458e-3, abs=0.0002e-3) # 1.1459e-3
    # standard uncertainty of the slope
    assert np.sqrt(b_cov[1, 1]) == approx(2.4161e-8, abs=0.0003e-8) # 2.4163e-3
    # covariance between intercept and slope
    assert b_cov[0, 1] == approx(-7.2747e-12, abs=0.0152e-12) # -7.2898e-12
    # residual
    assert np.sum(b_res*b_res) == approx(6.1697, abs=0.1253) # 6.0444
    # goodness of fit
    assert np.max(np.abs(b_res)) == approx(1.6322, abs=0.0057) # 1.6265
    # mixture no 1
    assert x[0] == approx(1.7004, abs=0.0001)
    assert np.sqrt(x_cov[0, 0]) == approx(2.0244e-3, abs=0.0003e-3) # 2.0241e-3
    # mixture no 2
    assert x[1] == approx(8.9863, abs=0.0005) # 8.9858
    assert np.sqrt(x_cov[1, 1]) == approx(9.9718e-3, abs=0.0003e-3) # 9.9720e-3

def test_example_2_second_order_poly():
    # See appendix B.2.2 Example 2, pages 26 - 28
    print('Example B LEAST 2\n')
    cal_data = b_read_cal_data(os.path.join(data_dir, 'b_least_2_data_cal.txt'))
    meas_data = b_read_meas_data(os.path.join(data_dir, 'b_least_2_data_meas.txt'))
    print('Second order polynomial\n')
    b, b_cov, b_res, x, x_cov = b_test(cal_data, meas_data, b_second_order_poly)
    # b0
    assert b[0] == approx(-1.4037e-4, abs=0.0927e-4) # -1.3110e-4
    assert np.sqrt(b_cov[0, 0]) == approx(1.175e-3, abs=0.001e-3)
    # b1
    assert b[1] == approx(2.4403e-5, abs=0.0002e-5) # 2.4401e-5
    assert np.sqrt(b_cov[1, 1]) == approx(5.901e-8, abs=0.001e-8)
    # b2
    assert b[2] == approx(-4.1096e-13, abs=0.0231e-13) # -4.0865e-13
    assert np.sqrt(b_cov[2, 2]) == approx(1.895e-13, abs=0.001e-13)
    # covariance between coefficients
    assert b_cov[0, 1] == approx(-2.057e-11, abs=0.002e-11)
    assert b_cov[0, 2] == approx(4.667e-17, abs=0.004e-17)
    assert b_cov[1, 2] == approx(-1.020e-20, abs=0.001e-20)
    # residual
    assert np.sum(b_res*b_res) == approx(1.4687, abs=0.0724) # 1.3963
    # goodness of fit
    assert np.max(np.abs(b_res)) == approx(0.8678, abs=0.0014) # 0.8664
    # mixture no 1
    assert x[0] == approx(1.7061, abs=0.0002)
    assert np.sqrt(x_cov[0, 0]) == approx(3.2910e-3, abs=0.0008e-3) # 3.2908e-3
    # mixture no 2
    assert x[1] == approx(8.9727, abs=0.0004) # 8.9723
    assert np.sqrt(x_cov[1, 1]) == approx(1.1762e-2, abs=0.0002e-2)
