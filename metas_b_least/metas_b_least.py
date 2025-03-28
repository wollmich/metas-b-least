# B_LEAST ISO 6143:2001
# Michael Wollensack METAS - 24.10.2024 - 28.03.2025

"""
METAS B LEAST is a Python implementation of the B LEAST program of the ISO 6143:2001 norm.
"""

import os
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def b_read_cal_data(filepath, delimiter='\t'):
    """
    Reads calibration data from a tab-separated text file.

    The file should have four columns:
    1. x values
    2. Standard uncertainties of x
    3. y values
    4. Standard uncertainties of y

    Parameters:
    filepath (str): The path to the file containing the calibration data.
    delimiter (str, optional): The delimiter used in the file. Defaults to '\\t'.

    Returns:
    numpy.ndarray: A 2D array where each row corresponds to a line in the file,
                   and each column corresponds to one of the four data types.

    Example:
    >>> cal_data = b_read_cal_data('cal_data.txt')
    >>> print(cal_data)
    [[1.0, 0.1, 2.0, 0.2],
     [2.0, 0.1, 4.0, 0.1],
     ...]
    """
    cal_data = np.loadtxt(filepath, delimiter=delimiter, ndmin=2)
    return cal_data

def b_read_meas_data(filepath, delimiter='\t'):
    """
    Reads measurement data from a tab-separated text file.

    The file should have two columns:
    1. y values
    2. Standard uncertainties of y

    Parameters:
    filepath (str): The path to the file containing the measurement data.
    delimiter (str, optional): The delimiter used in the file. Defaults to '\\t'.

    Returns:
    numpy.ndarray: A 2D array where each row corresponds to a line in the file,
                   and each column corresponds to one of the two data types.

    Example:
    >>> meas_data = b_read_meas_data('meas_data.txt')
    >>> print(meas_data)
    [[2.5, 0.1],
     [3.5, 0.2],
     ...]
    """
    meas_data = np.loadtxt(filepath, delimiter=delimiter, ndmin=2)
    return meas_data

def b_linear_func(y, b):
    '''
    Linear function
    '''
    x = b[0] + b[1]*y
    dx_dy = b[1] + 0*y
    dx_db0 = 1 + 0*y
    dx_db1 = y
    dx_db = [dx_db0, dx_db1]
    return [x, dx_dy, dx_db]

def b_second_order_poly(y, b):
    '''
    Second order polynomial
    '''
    x = b[0] + b[1]*y + b[2]*y**2
    dx_dy = b[1] + 2*b[2]*y
    dx_db0 = 1 + 0*y
    dx_db1 = y
    dx_db2 = y**2
    dx_db = [dx_db0, dx_db1, dx_db2]
    return [x, dx_dy, dx_db]

def b_third_order_poly(y, b):
    '''
    Third order polynomial
    '''
    x = b[0] + b[1]*y + b[2]*y**2 + b[3]*y**3
    dx_dy = b[1] + 2*b[2]*y + 3*b[3]*y**2
    dx_db0 = 1 + 0*y
    dx_db1 = y
    dx_db2 = y**2
    dx_db3 = y**3
    dx_db = [dx_db0, dx_db1, dx_db2, dx_db3]
    return [x, dx_dy, dx_db]

def b_power_func(y, b):
    '''
    Power function
    '''
    x = b[0] + b[1]*y**(1 + b[2])
    dx_dy = b[1]*(b[2] + 1)*y**b[2]
    dx_db0 = 1 + 0*y
    dx_db1 = y**(1 + b[2])
    dx_db2 = b[1]*y**(1 + b[2])*np.log(y)
    dx_db = [dx_db0, dx_db1, dx_db2]
    return [x, dx_dy, dx_db]

def b_exp_func(y, b):
    '''
    Expontential function
    '''
    x = b[0] + b[1]*np.exp(b[2]*y)
    dx_dy = b[1]*b[2]*np.exp(b[2]*y)
    dx_db0 = 1 + 0*y
    dx_db1 = np.exp(b[2]*y)
    dx_db2 = b[1]*y*np.exp(b[2]*y)
    dx_db = [dx_db0, dx_db1, dx_db2]
    return [x, dx_dy, dx_db]

def b_objective_func1(x, ux, y, uy, b, func):  # pylint: disable=R0913, R0917
    '''
    Computes the residuals for the x and y values and fit function

    Parameters:
    x (numpy.ndarray): A 1D array containing the x values.
    ux (numpy.ndarray): A 1D array containing the standard uncertainties of x.
    y (numpy.ndarray): A 1D array containing the y values.
    uy (numpy.ndarray): A 1D array containing the standard uncertainties of y.
    b (numpy.ndarray): A 1D array containing the coefficients b.
    func (callable): The fit function.

    Returns:
    numpy.ndarray: The residuals.

    Example:
    >>> x = np.array([1., 2.])
    >>> ux = np.array([0.1, 0.1])
    >>> y = np.array([2., 4.])
    >>> uy = np.array([0.2, 0.2])
    >>> b = np.array([0., 0.5])
    >>> b_objective_func1(x, ux, y, uy, b, b_linear_func)
    array([0., 0.])
    '''
    f = func(y, b)
    g = f[0] - x
    dg_dx = -1
    dg_dy = f[1]
    ug = np.sqrt((dg_dx*ux)**2 + (dg_dy*uy)**2)
    h = g/ug
    return h

def b_objective_func1c(cal_data, b, func):
    '''
    Computes the residuals for the given calibration data and fit function

    Parameters:
    cal_data (numpy.ndarray): A 2D array containing the calibration data.
    b (numpy.ndarray): A 1D array containing the coefficients b.
    func (callable): The fit function.

    Returns:
    numpy.ndarray: The residuals.

    Example:
    >>> cal_data = np.array([[1, 0.1, 2, 0.2], [2, 0.1, 4, 0.2]])
    >>> b = np.array([0., 0.5])
    >>> b_objective_func1c(cal_data, b, b_linear_func)
    array([0., 0.])
    '''
    x = cal_data[:, 0]
    ux = cal_data[:, 1]
    y = cal_data[:, 2]
    uy = cal_data[:, 3]
    h = b_objective_func1(x, ux, y, uy, b, func)
    return h

def b_objective_func2(x, ux, y, uy, y2, b, func):  # pylint: disable=R0913, R0917
    '''
    Computes the residuals for the x and y values and fit function

    Parameters:
    x (numpy.ndarray): A 1D array containing the x values.
    ux (numpy.ndarray): A 1D array containing the standard uncertainties of x.
    y (numpy.ndarray): A 1D array containing the y values.
    uy (numpy.ndarray): A 1D array containing the standard uncertainties of y.
    y2 (numpy.ndarray): A 1D array containing the y2 values.
    b (numpy.ndarray): A 1D array containing the coefficients b.
    func (callable): The fit function.

    Returns:
    numpy.ndarray: The residuals.

    Example:
    >>> x = np.array([1., 2.])
    >>> ux = np.array([0.1, 0.1])
    >>> y = np.array([2., 4.])
    >>> uy = np.array([0.2, 0.2])
    >>> y2 = np.array([2., 4.])
    >>> b = np.array([0., 0.5])
    >>> b_objective_func2(x, ux, y, uy, y2, b, b_linear_func)
    array([0., 0., 0., 0.])
    '''
    f = func(y2, b)
    x2 = f[0]
    wdx = (x2 - x)/ux
    wdy = (y2 - y)/uy
    g = np.concatenate((wdx, wdy))
    return g

def b_objective_func2c(cal_data, y2, b, func):
    '''
    Computes the residuals for the given calibration data and fit function

    Parameters:
    cal_data (numpy.ndarray): A 2D array containing the calibration data.
    y2 (numpy.ndarray): A 1D array containing the y2 values.
    b (numpy.ndarray): A 1D array containing the coefficients b.
    func (callable): The fit function.

    Returns:
    numpy.ndarray: The residuals.

    Example:
    >>> cal_data = np.array([[1, 0.1, 2, 0.2], [2, 0.1, 4, 0.2]])
    >>> y2 = np.array([2., 4.])
    >>> b = np.array([0., 0.5])
    >>> b_objective_func2c(cal_data, y2, b, b_linear_func)
    array([0., 0., 0., 0.])
    '''
    x = cal_data[:, 0]
    ux = cal_data[:, 1]
    y = cal_data[:, 2]
    uy = cal_data[:, 3]
    h = b_objective_func2(x, ux, y, uy, y2, b, func)
    return h

def b_covariance(cal_data, b, func):  # pylint: disable=R0914
    '''
    Computes the covariance matrix of the coefficients for the given
    calibration data and fit function.

    Parameters:
    cal_data (numpy.ndarray): A 2D array containing the calibration data.
    b (numpy.ndarray): A 1D array containing the coefficients b.
    func (callable): The fit function.

    Returns:
    numpy.ndarray: The covariance matrix of the coefficients b.

    Example:
    >>> cal_data = np.array([[1, 0.1, 2, 0.2], [2, 0.1, 4, 0.2]])
    >>> b = np.array([0., 0.5])
    >>> b_covariance(cal_data, b, b_linear_func)
    array([[0.1, -0.03], [-0.03, 0.01]])
    '''
    ux = cal_data[:, 1]
    y = cal_data[:, 2]
    uy = cal_data[:, 3]
    f = func(y, b)
    dg_dx = -1
    dg_dy = f[1]
    ug = np.sqrt((dg_dx*ux)**2 + (dg_dy*uy)**2)
    dh_db = np.array([dg_dbi/ug for dg_dbi in f[2]]).T
    db_dh = np.linalg.pinv(dh_db)
    dh_dx_ux = dg_dx/ug*ux
    dh_dy_uy = dg_dy/ug*uy
    dh_dx_ux_and_dh_dy_uy = np.concatenate((np.diag(dh_dx_ux), np.diag(dh_dy_uy)), axis=1)
    j = np.dot(db_dh, dh_dx_ux_and_dh_dy_uy)
    cv = np.dot(j, j.T)
    return cv

def b_least_start(cal_data, func):
    '''
    Computes the initial values of the coefficients for the fit function using the calibration data.

    Parameters:
    cal_data (numpy.ndarray): A 2D array containing the calibration data.
    func (callable): The fit function.

    Returns:
    numpy.ndarray: The initial coefficients b.

    Example:
    >>> cal_data = np.array([[1, 0.1, 2, 0.2], [2, 0.1, 4, 0.2]])
    >>> b_least_start(cal_data, b_linear_func)
    array([0., 0.5])
    '''
    x = cal_data[:, 0]
    y = cal_data[:, 2]
    if func is b_linear_func:
        b_start = np.flip(np.polyfit(y, x, 1))
    elif func is b_second_order_poly:
        b_start = np.flip(np.polyfit(y, x, 2))
    elif func is b_third_order_poly:
        b_start = np.flip(np.polyfit(y, x, 3))
    elif func is b_power_func:
        b0_b1 = np.flip(np.polyfit(y, x, 1))
        b_start = np.append(b0_b1, 0)
    elif func is b_exp_func:
        # x = b0 + b1*exp(b2*y)
        # x = b0 + b1*(1 + b2*y + b2^2/2*y^2 + ...)
        # x = b0 + b1 + b1*b2*y + b1*b2^2/2*y^2 + ...
        # x = c0 + c1*y + c2*y^2 + ...
        c = np.flip(np.polyfit(y, x, 2))
        b2 = 2*c[2]/c[1]
        b1 = c[1]/b2
        b0 = c[0] - b1
        b_start = np.array([b0, b1, b2])
    else:
        raise ValueError('Unknown fit function')
    return b_start

def _b_residuals1(params, cal_data, b_scale, func):
    b = params*b_scale
    f = b_objective_func1c(cal_data, b, func)
    #print(np.sum(f*f))
    return f

def _b_residuals2(params, cal_data, y2_b_scale, func):
    n = cal_data.shape[0]
    y2_b = params*y2_b_scale
    y2 = y2_b[:n]
    b = y2_b[n:]
    f = b_objective_func2c(cal_data, y2, b, func)
    #print(np.sum(f*f))
    return f

def _b_jacobian2(params, cal_data, y2_b_scale, func):  # pylint: disable=R0914
    n = cal_data.shape[0]
    nb = y2_b_scale.shape[0] - n
    y2_b = params*y2_b_scale
    y2 = y2_b[:n]
    b = y2_b[n:]
    y2_scale = y2_b_scale[:n]
    b_scale = y2_b_scale[n:]
    ux = cal_data[:, 1]
    uy = cal_data[:, 3]
    f = func(y2, b)
    dx2_dy2 = f[1]
    dx2_db = f[2]
    dy2_dy2 = 1
    jacobi = np.zeros((2*n, n + nb))
    for i in range(n):
        # Weighted x
        jacobi[i, i] = dx2_dy2[i] * y2_scale[i] / ux[i]
        for j in range(nb):
            jacobi[i, n+j] = dx2_db[j][i] * b_scale[j] / ux[i]
        # Weighted y
        jacobi[n+i, i] = dy2_dy2 * y2_scale[i] / uy[i]
    return jacobi

def b_least(cal_data, func):
    '''
    Fits the coefficients of the fit function using the calibration data.

    Parameters:
    cal_data (numpy.ndarray): A 2D array containing the calibration data.
    func (callable): The fit function.

    Returns:
    tuple: A tuple containing:
        - numpy.ndarray: The coefficients b.
        - numpy.ndarray: The covariance matrix of the coefficients b.
        - numpy.ndarray: The residuals.

    Example:
    >>> cal_data = np.array([[1, 0.1, 2, 0.2], [2, 0.1, 4, 0.2]])
    >>> b_least(cal_data, b_linear_func)
    (array([0., 0.5]), array([[0.1, -0.03], [-0.03, 0.01]]), array([0., 0., 0., 0.]))
    '''
    n = cal_data.shape[0]
    y2_start = cal_data[:, 2]
    b_start = b_least_start(cal_data, func)
    y2_b_start = np.concatenate((y2_start, b_start))
    y2_b_scale = np.copy(y2_b_start)
    y2_b_scale[y2_b_scale == 0] = 1
    y2_b_start2 = y2_b_start/y2_b_scale
    y2_b_lm = least_squares(_b_residuals2, y2_b_start2,
                            jac=_b_jacobian2, args=(cal_data, y2_b_scale, func),
                            method='lm')
    y2_b_opt = y2_b_lm.x*y2_b_scale
    y_opt = y2_b_opt[:n]
    b_opt = y2_b_opt[n:]
    b_opt_cov = b_covariance(cal_data, b_opt, func)
    b_res = b_objective_func2c(cal_data, y_opt, b_opt, func)
    return b_opt, b_opt_cov, b_res

def b_eval(meas_data, b, b_cov, func):
    '''
    Evaluates the fit function with the given coefficients at the measurement data.

    Parameters:
    meas_data (numpy.ndarray): A 2D array containing the measurement data.
    b (numpy.ndarray): A 1D array containing the coefficients b.
    b_cov (numpy.ndarray): A 2D array containing the covariance matrix of the coefficients b.
    func (callable): The fit function.

    Returns:
    tuple: A tuple containing:
        - numpy.ndarray: The x values.
        - numpy.ndarray: The covariance matrix of x.

    Example:
    >>> meas_data = np.array([[2.5, 0.1], [3.5, 0.2]])
    >>> b = np.array([0., 0.5])
    >>> b_cov = np.array([[0.1, -0.03], [-0.03, 0.01]])
    >>> b_eval(meas_data, b, b_cov, b_linear_func)
    (array([1.25, 1.75]), array([[0.015, 0.0075], [0.0075, 0.0225]]))
    '''
    x, _, x_cov, _, _ = b_eval_xy(meas_data, b, b_cov, func)
    return x, x_cov

def b_eval_xy(meas_data, b, b_cov, func):  # pylint: disable=R0914
    '''
    Evaluates the fit function with the given coefficients at the measurement data.

    Parameters:
    meas_data (numpy.ndarray): A 2D array containing the measurement data.
    b (numpy.ndarray): A 1D array containing the coefficients b.
    b_cov (numpy.ndarray): A 2D array containing the covariance matrix of the coefficients b.
    func (callable): The fit function.

    Returns:
    tuple: A tuple containing:
        - numpy.ndarray: The x values.
        - numpy.ndarray: The y values.
        - numpy.ndarray: The covariance matrix of x.
        - numpy.ndarray: The covariance matrix of y.
        - numpy.ndarray: The covariance matrix between x and y.

    Example:
    >>> meas_data = np.array([[2.5, 0.1], [3.5, 0.2]])
    >>> b = np.array([0., 0.5])
    >>> b_cov = np.array([[0.1, -0.03], [-0.03, 0.01]])
    >>> b_eval_xy(meas_data, b, b_cov, b_linear_func)
    (array([1.25, 1.75]), array([2.5, 3.5]), 
     array([[0.015, 0.0075], [0.0075, 0.0225]], 
     array([[0.01, 0.], [0., 0.04]]),
     array([[0.005, 0.], [0., 0.02 ]]))
    '''
    y = meas_data[:, 0]
    uy = meas_data[:, 1]
    ny = y.size
    nb = b.size
    f = func(y, b)
    x = f[0]
    dx_dy = f[1]
    dx_db = np.array(f[2]).T
    jx = np.concatenate((np.diag(dx_dy), dx_db), axis=1)
    jy = np.concatenate((np.eye(ny), np.zeros_like(dx_db)), axis=1)
    j = np.concatenate((jx, jy), axis=0)
    y_cov = np.diag(uy**2)
    cv_in = np.zeros((ny + nb, ny + nb))
    cv_in[:ny, :ny] = y_cov
    cv_in[ny:, ny:] = b_cov
    cov = np.dot(np.dot(j, cv_in), j.T)
    x_cov = cov[:ny, :ny]
    y_cov = cov[ny:, ny:]
    xy_cov = cov[:ny, ny:]
    return x, y, x_cov, y_cov, xy_cov

def b_disp_cal_data(cal_data):
    '''
    Displays the calibration data.

    Parameters:
    cal_data (numpy.ndarray): A 2D array containing the calibration data.

    Returns:
    None
    '''
    print('Calibration data:')
    print(cal_data)

def b_disp_cal_results(b, b_cov, b_res):
    '''
    Displays the coefficients, their uncertainties, the covariance matrix,
    the residual and the maximum absolute value of weighted deviations.

    Parameters:
    b (numpy.ndarray): A 1D array containing the coefficients b.
    b_cov (numpy.ndarray): A 2D array containing the covariance matrix of the coefficients b.
    b_res (numpy.ndarray): A 1D array containing the residuals.

    Returns:
    None
    '''
    print('Coefficients b')
    print(b)
    print('Uncertainties u(b)')
    print(np.sqrt(np.diag(b_cov)))
    print('Covariance cov(b)')
    print(b_cov)
    print('Residual')
    print(np.sum(b_res*b_res))
    print('Maximum absolute value of weighted deviations')
    print(np.max(np.abs(b_res)))

def b_disp_meas_results(x, x_cov, meas_data):
    '''
    Displays the measurement data.

    Parameters:
    x (numpy.ndarray): A 1D array containing the x values.
    x_cov (numpy.ndarray): A 2D array containing the covariance matrix of x.
    meas_data (numpy.ndarray): A 2D array containing the measurement data.

    Returns:
    None
    '''
    print('Measurement data:')
    ux = np.sqrt(np.diag(x_cov))
    print(np.concatenate((np.array([x, ux]).T, meas_data), axis=1))
    if ux.size > 1:
        print('Covariance cov(x)')
        print(x_cov)

def b_plot(cal_data, meas_data, b, b_cov, func):  # pylint: disable=R0914
    '''
    Plots the calibration data, the measurement data and the fit function using the coefficients.

    Parameters:
    cal_data (numpy.ndarray): A 2D array containing the calibration data.
    meas_data (numpy.ndarray): A 2D array containing the measurement data.
    b (numpy.ndarray): A 1D array containing the coefficients b.
    b_cov (numpy.ndarray): A 2D array containing the covariance matrix of the coefficients b.
    func (callable): The fit function.

    Returns:
    None
    '''
    k = 2
    # figure
    _, ax = plt.subplots()
    # fit function
    ymin = np.min(np.array([np.min(cal_data[:,2]), np.min(meas_data[:,0])]))
    ymax = np.max(np.array([np.max(cal_data[:,2]), np.max(meas_data[:,0])]))
    fy = np.linspace(ymin, ymax, num=100).reshape(-1, 1)
    f_data = np.concatenate((fy, np.zeros_like(fy)), axis=1)
    fx, fx_cov = b_eval(f_data, b, b_cov, func)
    ufx = np.sqrt(np.diag(fx_cov))
    ax.plot(fx, fy, color='blue', label='Fit x = f(y)')
    ax.fill_betweenx(fy.flatten(), (fx-k*ufx).flatten(), (fx+k*ufx).flatten(),
                     color='blue', alpha=0.5)
    # calibration data
    ax.errorbar(cal_data[:,0], cal_data[:,2], xerr=k*cal_data[:,1], yerr=k*cal_data[:,3],
                fmt='.', color='red', ecolor='red', capsize=3, label='Reference points')
    for i in range(cal_data.shape[0]):
        _b_plot_ellipse(ax, cal_data[i,0], cal_data[i,2], np.array([[cal_data[i,1]**2, 0],
                                                                    [0, cal_data[i,3]**2]]), 'red')
    # measurement data
    x, y, x_cov, y_cov, xy_cov = b_eval_xy(meas_data, b, b_cov, func)
    ax.errorbar(x, meas_data[:,0], xerr=k*np.sqrt(np.diag(x_cov)), yerr=k*np.sqrt(np.diag(y_cov)),
                fmt='.', color='black', ecolor='black', capsize=3, label='Measurement points')
    for i in range(meas_data.shape[0]):
        _b_plot_ellipse(ax, x[i], y[i], np.array([[x_cov[i,i], xy_cov[i,i]],
                                                  [xy_cov[i,i], y_cov[i,i]]]), 'black')
    plt.xlabel('Assigned value x')
    plt.ylabel('Instrument response y')
    plt.legend()
    plt.show()

def _b_plot_ellipse(ax, px, py, cv, color):
    k = 2.45
    d, v = np.linalg.eig(k**2*cv)
    t = np.linspace(0, 2*np.pi, num=100)
    e = np.dot(v, np.diag(np.sqrt(d)))
    f = np.array([np.cos(t), np.sin(t)])
    g = np.dot(e, f)
    ax.fill((g[0,:] + px).flatten(), (g[1,:] + py).flatten(), color=color, alpha=0.5)

def b_test(cal_data, meas_data, func):
    '''
    Fits the coefficients of the fit function using the calibration data
    and evaluates the fit function at the measurement data.

    Parameters:
    cal_data (numpy.ndarray): A 2D array containing the calibration data.
    meas_data (numpy.ndarray): A 2D array containing the measurement data.
    func (callable): The fit function.

    Returns:
    tuple: A tuple containing:
        - numpy.ndarray: The coefficients b.
        - numpy.ndarray: The covariance matrix of the coefficients b.
        - numpy.ndarray: The residuals.
        - numpy.ndarray: The x values.
        - numpy.ndarray: The covariance matrix of x.
    
    Example:
    >>> cal_data = np.array([[1, 0.1, 2, 0.2], [2, 0.1, 4, 0.2]])
    >>> meas_data = np.array([[2.5, 0.1], [3.5, 0.2]])
    >>> b, b_cov, b_res, x, x_cov = b_test(cal_data, meas_data, b_linear_func)
    '''
    b_disp_cal_data(cal_data)
    b, b_cov, b_res = b_least(cal_data, func)
    b_disp_cal_results(b, b_cov, b_res)
    x, x_cov = b_eval(meas_data, b, b_cov, func)
    b_disp_meas_results(x, x_cov, meas_data)
    return b, b_cov, b_res, x, x_cov

# Monte Carlo

def b_sample_cal_data_mc(cal_data, nsamples=10000, seed=None):
    '''
    Generates samples from the calibration data using a Monte Carlo method.

    Parameters:
    cal_data (numpy.ndarray): A 2D array containing the calibration data.
                              Each row should contain [x, std_x, y, std_y].
    nsamples (int, optional): Number of samples to generate. Defaults to 10000.
    seed (int, optional): Seed for the random number generator. Defaults to None.

    Returns:
    numpy.ndarray: A 3D array containing the calibration samples.
                   The shape of the array is (nsamples, n, 2), where n is the number of data points.

    Example:
    >>> cal_data = np.array([[1, 0.1, 2, 0.2], [2, 0.1, 4, 0.2]])
    >>> cal_samples = b_sample_cal_data_mc(cal_data, nsamples=5, seed=42)
    '''
    if seed is not None:
        np.random.seed(seed)
    n = cal_data.shape[0]
    cal_samples = np.zeros((nsamples, n, 2))
    for j in range(n):
        cal_samples[:, j, 0] = np.random.normal(cal_data[j, 0], cal_data[j, 1], nsamples)
        cal_samples[:, j, 1] = np.random.normal(cal_data[j, 2], cal_data[j, 3], nsamples)
    return cal_samples

def b_sample_meas_data_mc(meas_data, nsamples=10000, seed=None):
    '''
    Generates samples from the measurement data using a Monte Carlo method.

    Parameters:
    meas_data (numpy.ndarray): A 2D array containing the measurement data.
                               Each row should contain [y, std_y].
    nsamples (int, optional): Number of samples to generate. Defaults to 10000.
    seed (int, optional): Seed for the random number generator. Defaults to None.

    Returns:
    numpy.ndarray: A 2D array containing the measurement samples.
                   The shape of the array is (nsamples, n), where n is the number of data points.

    Example:
    >>> meas_data = np.array([[2.5, 0.1], [3.5, 0.2]])
    >>> meas_samples = b_sample_meas_data_mc(meas_data, nsamples=5, seed=43)
    '''
    if seed is not None:
        np.random.seed(seed)
    n = meas_data.shape[0]
    meas_samples = np.zeros((nsamples, n))
    for j in range(n):
        meas_samples[:, j] = np.random.normal(meas_data[j, 0], meas_data[j, 1], nsamples)
    return meas_samples

def b_least_mc(cal_samples, func):  # pylint: disable=R0914
    '''
    Fits the coefficients of the fit function for each sample using the calibration samples.

    Parameters:
    cal_samples (numpy.ndarray): A 3D array containing the calibration samples.
    func (callable): The fit function.

    Returns:
    numpy.ndarray: A 2D array where each row contains the coefficients b for one sample.
    '''
    nsamples = cal_samples.shape[0]
    cal_data = _b_cal_samples_to_cal_data(cal_samples)
    n = cal_data.shape[0]
    y2_start = cal_data[:, 2]
    cal_data_i = np.copy(cal_data)
    b_start = b_least_start(cal_data, func)
    y2_b_start = np.concatenate((y2_start, b_start))
    y2_b_scale = np.copy(y2_b_start)
    y2_b_scale[y2_b_scale == 0] = 1
    y2_b_start2 = y2_b_start/y2_b_scale
    b_samples = np.zeros((nsamples, b_start.size))
    for i in range(nsamples):
        cal_data_i[:, 0] = cal_samples[i, :, 0]
        cal_data_i[:, 2] = cal_samples[i, :, 1]
        y2_b_i_lm = least_squares(_b_residuals2, y2_b_start2, args=(cal_data_i, y2_b_scale, func),
                                  method='lm')
        y2_b_opt_i = y2_b_i_lm.x*y2_b_scale
        b_opt_i = y2_b_opt_i[n:]
        b_samples[i, :] = b_opt_i
    return b_samples

def b_eval_mc(meas_samples, b_samples, func):
    '''
    Evaluates the fit function with the given coefficients at the measurement samples.

    Parameters:
    meas_samples (numpy.ndarray): A 2D array containing the measurement samples.
    b_samples (numpy.ndarray): A 2D array where each row contains the coefficients b for one sample.
    func (callable): The fit function.

    Returns:
    - numpy.ndarray: A 2D array where each row contains the x values for one sample.
    '''
    nsamples = meas_samples.shape[0]
    nmeas = meas_samples.shape[1]
    x_samples = np.zeros((nsamples, nmeas))
    for i in range(nsamples):
        x_samples[i, :] = func(meas_samples[i, :], b_samples[i, :])[0]
    return x_samples

def b_mean_cov_mc(samples):
    '''
    Computes the mean values and covariance matrix of the samples.

    Parameters:
    samples (numpy.ndarray): A 2D array where each row contains one sample.

    Returns:
    tuple: A tuple containing:
        - numpy.ndarray: The mean values of the samples.
        - numpy.ndarray: The covariance matrix of the samples.

    Example:
    >>> samples = np.random.rand(10, 3)  # 10 samples, each with 3 variables
    >>> mean, covariance = b_mean_cov_mc(samples)
    '''
    m = np.mean(samples, axis=0)
    cov = np.asmatrix(np.cov(samples, rowvar=False))
    return m, cov

def _b_cal_samples_to_cal_data(cal_samples):
    cal_data = np.zeros((cal_samples.shape[1], 4))
    cal_data[:, 0] = np.mean(cal_samples[:, :, 0], axis=0)
    cal_data[:, 1] = np.std(cal_samples[:, :, 0], axis=0)
    cal_data[:, 2] = np.mean(cal_samples[:, :, 1], axis=0)
    cal_data[:, 3] = np.std(cal_samples[:, :, 1], axis=0)
    return cal_data

def b_disp_cal_data_mc(cal_samples):
    '''
    Displays the calibration data.

    Parameters:
    cal_samples (numpy.ndarray): A 3D array containing the calibration samples.

    Returns:
    None
    '''
    cal_data = _b_cal_samples_to_cal_data(cal_samples)
    b_disp_cal_data(cal_data)

def b_disp_cal_results_mc(b_samples):
    '''
    Displays the coefficients, their uncertainties, the covariance matrix.

    Parameters:
    b_samples (numpy.ndarray): A 2D array where each row contains the coefficients b
                               for one samples.

    Returns:
    None
    '''
    b, b_cov = b_mean_cov_mc(b_samples)
    b_res = np.nan
    b_disp_cal_results(b, b_cov, b_res)

def b_disp_meas_results_mc(x_samples, meas_samples):
    '''
    Displays the measurement data.

    Parameters:
    x_samples (numpy.ndarray): A 2D array where each row contains the x values for one sample.
    meas_samples (numpy.ndarray): A 2D array containing the measurement samples.

    Returns:
    None
    '''
    x, x_cov = b_mean_cov_mc(x_samples)
    meas_data = np.zeros((meas_samples.shape[1], 2))
    meas_data[:, 0] = np.mean(meas_samples, axis=0)
    meas_data[:, 1] = np.std(meas_samples, axis=0)
    b_disp_meas_results(x, x_cov, meas_data)

def b_test_mc(cal_data, meas_data, func, nsamples=10000):
    '''
    Fits the coefficients of the fit function using the calibration data
    and evaluates the fit function at the measurement data.

    Parameters:
    cal_data (numpy.ndarray): A 2D array containing the calibration data.
    meas_data (numpy.ndarray): A 2D array containing the measurement data.
    func (callable): The fit function.
    nsamples (int, optional): Number of samples to generate. Defaults to 10000.

    Returns:
    tuple: A tuple containing:
        - numpy.ndarray: A 2D array where each row contains the coefficients b for one sample.
        - numpy.ndarray: A 2D array where each row contains the x values for one sample.
    
    Example:
    >>> cal_data = np.array([[1, 0.1, 2, 0.2], [2, 0.1, 4, 0.2]])
    >>> meas_data = np.array([[2.5, 0.1], [3.5, 0.2]])
    >>> b_samples, x_samples = b_test_mc(cal_data, meas_data, b_linear_func)
    '''
    cal_samples = b_sample_cal_data_mc(cal_data, nsamples)
    meas_samples = b_sample_meas_data_mc(meas_data, nsamples)
    b_disp_cal_data_mc(cal_samples)
    b_samples = b_least_mc(cal_samples, func)
    b_disp_cal_results_mc(b_samples)
    x_samples = b_eval_mc(meas_samples, b_samples, func)
    b_disp_meas_results_mc(x_samples, meas_samples)
    return b_samples, x_samples

# Examples

def b_example_1():
    '''
    Example B LEAST 1
    '''
    print('Example B LEAST 1\n')
    cal_data = b_read_cal_data(os.path.join(data_dir, 'b_least_1_data_cal.txt'))
    meas_data = b_read_meas_data(os.path.join(data_dir, 'b_least_1_data_meas.txt'))
    print('Linear function\n')
    b_test(cal_data, meas_data, b_linear_func)

def b_example_2():
    '''
    Example B LEAST 2
    '''
    print('Example B LEAST 2\n')
    cal_data = b_read_cal_data(os.path.join(data_dir, 'b_least_2_data_cal.txt'))
    meas_data = b_read_meas_data(os.path.join(data_dir, 'b_least_2_data_meas.txt'))
    print('Linear function\n')
    b_test(cal_data, meas_data, b_linear_func)
    print('Second order polynomial\n')
    b_test(cal_data, meas_data, b_second_order_poly)

def b_example_3():
    '''
    Example B LEAST 3
    '''
    print('Example B LEAST 3\n')
    cal_data = b_read_cal_data(os.path.join(data_dir, 'b_least_3_data_cal.txt'))
    meas_data = b_read_meas_data(os.path.join(data_dir, 'b_least_3_data_meas.txt'))
    print('Linear function\n')
    b_test(cal_data, meas_data, b_linear_func)
    print('Power function\n')
    b_test(cal_data, meas_data, b_power_func)
    print('Exponential function\n')
    b_test(cal_data, meas_data, b_exp_func)

def b_example_mc_1():
    '''
    Example B LEAST Monte Carlo 1
    '''
    print('Example B LEAST Monte Carlo 1\n')
    cal_data = b_read_cal_data(os.path.join(data_dir, 'b_least_1_data_cal.txt'))
    meas_data = b_read_meas_data(os.path.join(data_dir, 'b_least_1_data_meas.txt'))
    print('Linear function\n')
    b_test_mc(cal_data, meas_data, b_linear_func)

def b_example_mc_2():
    '''
    Example B LEAST Monte Carlo 2
    '''
    print('Example B LEAST Monte Carlo 2\n')
    cal_data = b_read_cal_data(os.path.join(data_dir, 'b_least_2_data_cal.txt'))
    meas_data = b_read_meas_data(os.path.join(data_dir, 'b_least_2_data_meas.txt'))
    print('Linear function\n')
    b_test_mc(cal_data, meas_data, b_linear_func)
    print('Second order polynomial\n')
    b_test_mc(cal_data, meas_data, b_second_order_poly)

def b_example_mc_3():
    '''
    Example B LEAST Monte Carlo 3
    '''
    print('Example B LEAST Monte Carlo 3\n')
    cal_data = b_read_cal_data(os.path.join(data_dir, 'b_least_3_data_cal.txt'))
    meas_data = b_read_meas_data(os.path.join(data_dir, 'b_least_3_data_meas.txt'))
    print('Linear function\n')
    b_test_mc(cal_data, meas_data, b_linear_func)
    print('Power function\n')
    b_test_mc(cal_data, meas_data, b_power_func)
    print('Exponential function\n')
    b_test_mc(cal_data, meas_data, b_exp_func)

if __name__ == "__main__":
    b_example_1()
    b_example_2()
    b_example_3()
    b_example_mc_1()
    b_example_mc_2()
    b_example_mc_3()
