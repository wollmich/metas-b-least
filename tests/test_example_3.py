# B_LEAST ISO 6143:2001
# Michael Wollensack METAS - 24.10.2024 - 03.12.2024

# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=wildcard-import
# pylint: disable=unused-wildcard-import

from pytest import approx
from metas_b_least import *

def test_example_3_linear_func():
    # See appendix B.2.3 Example 3, pages 28 - 30
    print('Example B LEAST 3\n')
    cal_data = b_read_cal_data(os.path.join(data_dir, 'b_least_3_data_cal.txt'))
    meas_data = b_read_meas_data(os.path.join(data_dir, 'b_least_3_data_meas.txt'))
    print('Linear function\n')
    _, _, b_res, _, _ = b_test(cal_data, meas_data, b_linear_func)
    # residual
    assert np.sum(b_res*b_res) == approx(272.6392, abs=0.0001)
    # goodness of fit
    assert np.max(np.abs(b_res)) == approx(6.8352, abs=0.0010) # 6.8361

def test_example_3_power_func():
    # See appendix B.2.3 Example 3, pages 28 - 30
    print('Example B LEAST 3\n')
    cal_data = b_read_cal_data(os.path.join(data_dir, 'b_least_3_data_cal.txt'))
    meas_data = b_read_meas_data(os.path.join(data_dir, 'b_least_3_data_meas.txt'))
    print('Power function\n')
    b, b_cov, b_res, x, x_cov = b_test(cal_data, meas_data, b_power_func)
    # b0
    assert b[0] == approx(1.2128e-1, abs=0.0002e-1)
    assert np.sqrt(b_cov[0, 0]) == approx(1.7821e-2, abs=0.0536e-2) # 1.8356e-2
    # b1
    assert b[1] == approx(5.1213e-4, abs=0.0003e-4)
    assert np.sqrt(b_cov[1, 1]) == approx(2.3693e-5, abs=0.0799e-5) # 2.4491e-5
    # b2
    assert b[2] == approx(8.4986e-2, abs=0.0005e-2)
    assert np.sqrt(b_cov[2, 2]) == approx(4.9745e-3, abs=0.1668e-3) # 5.1412e-3
    # covariance between coefficients
    assert b_cov[0, 1] == approx(-3.8265e-7, abs=0.2731e-7) # -4.0995e-7
    assert b_cov[0, 2] == approx(7.9326e-5, abs=0.5720e-5) # 8.5045e-5
    assert b_cov[1, 2] == approx(-1.1780e-7, abs=0.0806e-7) # -1.2585e-7
    # residual
    assert np.sum(b_res*b_res) == approx(8.3804, abs=0.0001)
    # goodness of fit
    assert np.max(np.abs(b_res)) == approx(1.1594, abs=0.0001)
    # mixture no 1
    assert x[0] == approx(5.3456, abs=0.0001)
    assert np.sqrt(x_cov[0, 0]) == approx(1.4141e-2, abs=0.0049e-2) # 1.4189e-2

def test_example_3_exp_func():
    # See appendix B.2.3 Example 3, pages 28 - 30
    # Uncertainties are not compared, see
    # https://github.com/wollmich/metas-b-least/issues/1
    print('Example B LEAST 3\n')
    cal_data = b_read_cal_data(os.path.join(data_dir, 'b_least_3_data_cal.txt'))
    meas_data = b_read_meas_data(os.path.join(data_dir, 'b_least_3_data_meas.txt'))
    print('Exponential function\n')
    b, _, b_res, x, _ = b_test(cal_data, meas_data, b_exp_func)
    # b0
    assert b[0] == approx(-4.8019e1, abs=0.0577e1) # -4.7962e1
    # b1
    assert b[1] == approx(4.8024e1, abs=0.0556e1) # 4.7968e1
    # b2
    assert b[2] == approx(2.1261e-5, abs=0.0001e-2)
    # residual
    assert np.sum(b_res*b_res) == approx(0.6581, abs=0.0009) # 0.6572
    # goodness of fit
    assert np.max(np.abs(b_res)) == approx(0.3552, abs=0.0023) # 0.3529
    # mixture no 1
    assert x[0] == approx(5.3357, abs=0.0001)
