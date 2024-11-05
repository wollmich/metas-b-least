# METAS B LEAST

METAS B LEAST is a Python implementation of the B LEAST program of the [ISO 6143:2001](https://www.iso.org/standard/24665.html) norm.
The derivations of the different fit functions have been explicitly programmed, see [metas_b_lest.py](https://github.com/wollmich/metas-b-least/blob/main/metas_b_least/metas_b_least.py).
The program has been verified against [METAS UncLib](https://www.metas.ch/unclib) which is using automatic differentiation.

The following link will launch an interactive Python environment where you can you use METAS B LEAST:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/wollmich/metas-b-least/HEAD)

## Examples

Take a look at the following code example for the usage of the METAS B LEAST Python package:

```python
from metas_b_least import *

# Calibration and measurement data
cal_data = b_read_cal_data(os.path.join(data_dir, 'b_least_1_data_cal.txt'))
meas_data = b_read_meas_data(os.path.join(data_dir, 'b_least_1_data_meas.txt'))
b_disp_cal_data(cal_data)

# Fit coefficients of the fit function using the calibration data
b, b_cov, b_res = b_least(cal_data, b_linear_func)
b_disp_cal_results(b, b_cov, b_res)

# Evaluate the fit function with the coefficients at the measurement data
x, x_cov = b_eval(meas_data, b, b_cov, b_linear_func)
b_disp_meas_results(x, x_cov, meas_data)

# Plot calibration data, measurement data and fit function
b_plot(cal_data, meas_data, b, b_cov, b_linear_func)
```

See as well the following Jupyter Notebooks:

- [Example 1](https://github.com/wollmich/metas-b-least/blob/main/metas_b_least/Example_B_LEAST_1.ipynb)
- [Example 2](https://github.com/wollmich/metas-b-least/blob/main/metas_b_least/Example_B_LEAST_2.ipynb)
- [Example 3](https://github.com/wollmich/metas-b-least/blob/main/metas_b_least/Example_B_LEAST_3.ipynb)

## Functions

### Input Functions

**b_read_cal_data** reads calibration data from tabular separated text file where the first column are the `x` values, the second column are the standard uncertainties of `x`, the third column are the `y` values and the forth column are the standard uncertainties of `y`.

**b_read_meas_data** reads measurement data from tabular separated text file where the first column are the `y` values and the second column are the standard uncertainties of `y`.

### Processing Functions

**b_least** fits the coefficients `b` of the fit function `func` using the calibration data `cal_data`.

**b_eval** evaluates the fit function `func` with the coefficients `b` at the measurement data `meas_data`.

The following fit functions are available:

| Name                    | Function                             |
|:------------------------|:-------------------------------------|
| **b_linear_func**       | $$x = b_0 + b_1y$$                   |
| **b_second_order_poly** | $$x = b_0 + b_1y + b_2y^2$$          |
| **b_third_order_poly**  | $$x = b_0 + b_1y + b_2y^2 + b_3y^3$$ |
| **b_power_func**        | $$x = b_0 + b_1y^{(1 + b_2)}$$       |
| **b_exp_func**          | $$x = b_0 + b_1e^{b_2y}$$            |

### Output Functions

**b_disp_cal_data** displays the calibration data `cal_data`.

**b_disp_cal_results** displays the coefficients `b`, the uncertainties of `b`, the covariance matrix of `b`, the residual and the maximum absolute value of weighted deviations.

**b_disp_meas_results** displays the measurement data `x` and `meas_data`.

**b_plot** plots the calibration data `cal_data`, the measurement data `meas_data` and the fit function using the coefficients `b`.

## Source Code

https://github.com/wollmich/metas-b-least/

## Releases

https://pypi.org/project/metas-b-least/

## Requirements

- [NumPy](https://pypi.org/project/numpy/)
- [SciPy](https://pypi.org/project/scipy/)
- [Matplotlib](https://pypi.org/project/matplotlib/)

---

Michael Wollensack METAS - 05.11.2024
