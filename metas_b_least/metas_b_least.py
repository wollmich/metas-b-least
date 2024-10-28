# B_LEAST ISO 6143:2001
# Michael Wollensack METAS - 24.10.2024 - 28.10.2024

import os
import numpy as np
from scipy.optimize import least_squares

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def b_read_cal_data(filepath, delimiter='\t'):
	"""
	b_read_cal_data reads calibration data from tabular separated text file
	where the first column are the x values, the second column are the
	standard uncertainties of x, the third column are the y values and the
	forth column are the standard uncertainties of y.
	"""
	cal_data = np.loadtxt(filepath, delimiter=delimiter, ndmin=2)
	return cal_data

def b_read_meas_data(filepath, delimiter='\t'):
	"""
	b_read_meas_data reads measurement data from tabular separated text file
	where the first column are the y values and the second column are the
	standard uncertainties of y.
	"""
	meas_data = np.loadtxt(filepath, delimiter=delimiter, ndmin=2)
	return meas_data

def b_linear_func(y, b):
	x = b[0] + b[1]*y
	dx_dy = b[1] + 0*y
	dx_db0 = 1 + 0*y
	dx_db1 = y
	dx_db = [dx_db0, dx_db1]
	return [x, dx_dy, dx_db]

def b_second_order_poly(y, b):
	x = b[0] + b[1]*y + b[2]*y**2
	dx_dy = b[1] + 2*b[2]*y
	dx_db0 = 1 + 0*y
	dx_db1 = y
	dx_db2 = y**2
	dx_db = [dx_db0, dx_db1, dx_db2]
	return [x, dx_dy, dx_db]

def b_third_order_poly(y, b):
	x = b[0] + b[1]*y + b[2]*y**2 + b[3]*y**3
	dx_dy = b[1] + 2*b[2]*y + 3*b[3]*y**2
	dx_db0 = 1 + 0*y
	dx_db1 = y
	dx_db2 = y**2
	dx_db3 = y**3
	dx_db = [dx_db0, dx_db1, dx_db2, dx_db3]
	return [x, dx_dy, dx_db]

def b_power_func(y, b):
	x = b[0] + b[1]*y**(1 + b[2])
	dx_dy = b[1]*(b[2] + 1)*y**b[2]
	dx_db0 = 1 + 0*y
	dx_db1 = y**(1 + b[2])
	dx_db2 = b[1]*y**(1 + b[2])*np.log(y)
	dx_db = [dx_db0, dx_db1, dx_db2]
	return [x, dx_dy, dx_db]

def b_exp_func(y, b):
	x = b[0] + b[1]*np.exp(b[2]*y)
	dx_dy = b[1]*b[2]*np.exp(b[2]*y)
	dx_db0 = 1 + 0*y
	dx_db1 = np.exp(b[2]*y)
	dx_db2 = b[1]*y*np.exp(b[2]*y)
	dx_db = [dx_db0, dx_db1, dx_db2]
	return [x, dx_dy, dx_db]

def b_objective_func(x, ux, y, uy, b, func):
	f = func(y, b)
	g = f[0] - x
	dg_dx = -1
	dg_dy = f[1]
	ug = np.sqrt((dg_dx*ux)**2 + (dg_dy*uy)**2)
	h = g/ug
	return h

def b_objective_func2(cal_data, b, func):
	x = cal_data[:, 0]
	ux = cal_data[:, 1]
	y = cal_data[:, 2]
	uy = cal_data[:, 3]
	h = b_objective_func(x, ux, y, uy, b, func)
	return h

def b_covariance(cal_data, b, func):
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
	x = cal_data[:, 0]
	y = cal_data[:, 2]
	if func == b_linear_func:
		b_start = np.flip(np.polyfit(y, x, 1))
	elif func == b_second_order_poly:
		b_start = np.flip(np.polyfit(y, x, 2))
	elif func == b_third_order_poly:
		b_start = np.flip(np.polyfit(y, x, 3))
	elif func == b_power_func:
		b0_b1 = np.flip(np.polyfit(y, x, 1))
		b_start = np.append(b0_b1, 0)
	elif func == b_exp_func:
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

def _b_residuals(params, cal_data, b_scale, func):
	b = params*b_scale
	f = b_objective_func2(cal_data, b, func)
	#print(np.sum(f*f))
	return f

def b_least(cal_data, func):
	'''
	b_least fits the coefficients b of the fit function func using the
	calibration data cal_data.
	'''
	b_start = b_least_start(cal_data, func)
	b_scale = np.copy(b_start)
	b_scale[b_scale == 0] = 1
	b_start2 = b_start/b_scale
	b_lm = least_squares(_b_residuals, b_start2, args=(cal_data, b_scale, func), method='lm')
	b_opt = b_lm.x*b_scale
	b_opt_cov = b_covariance(cal_data, b_opt, func)
	b_res = b_objective_func2(cal_data, b_opt, func)
	return b_opt, b_opt_cov, b_res

def b_eval(meas_data, b, b_cov, func):
	'''
	b_eval evaluates the fit function func with the coefficients b at the 
	measurement data meas_data.
	'''
	y = meas_data[:, 0]
	uy = meas_data[:, 1]
	f = func(y, b)
	x = f[0]
	dx_dy = f[1]
	dx_db = np.array(f[2]).T
	j = np.concatenate((np.diag(dx_dy), dx_db), axis=1)
	y_cov = np.diag(uy**2)
	ny = y.size
	nb = b.size
	cv_in = np.zeros((ny + nb, ny + nb))
	cv_in[:ny, :ny] = y_cov
	cv_in[ny:, ny:] = b_cov
	x_cov = np.dot(np.dot(j, cv_in), j.T)
	return x, x_cov

def b_disp_cal_data(cal_data):
	print('Calibration data:')
	print(cal_data)

def b_disp_cal_results(b, b_cov, b_res):
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
	print('Measurement data:')
	ux = np.sqrt(np.diag(x_cov))
	print(np.concatenate((np.array([x, ux]).T, meas_data), axis=1))
	if (ux.size > 1):
		print('Covariance cov(x)')
		print(x_cov)

def b_test(cal_data, meas_data, func):
	b_disp_cal_data(cal_data)
	b, b_cov, b_res = b_least(cal_data, func)
	b_disp_cal_results(b, b_cov, b_res)
	x, x_cov = b_eval(meas_data, b, b_cov, func)
	b_disp_meas_results(x, x_cov, meas_data)
	return b, b_cov, b_res, x

def b_example_1():
	print('Example B LEAST 1\n')
	cal_data = b_read_cal_data(os.path.join(data_dir, 'b_least_1_data_cal.txt'))
	meas_data = b_read_meas_data(os.path.join(data_dir, 'b_least_1_data_meas.txt'))
	print('Linear function\n')
	b_test(cal_data, meas_data, b_linear_func)

def b_example_2():
	print('Example B LEAST 2\n')
	cal_data = b_read_cal_data(os.path.join(data_dir, 'b_least_2_data_cal.txt'))
	meas_data = b_read_meas_data(os.path.join(data_dir, 'b_least_2_data_meas.txt'))
	print('Linear function\n')
	b_test(cal_data, meas_data, b_linear_func)
	print('Second order polynomial\n')
	b_test(cal_data, meas_data, b_second_order_poly)

def b_example_3():
	print('Example B LEAST 3\n')
	cal_data = b_read_cal_data(os.path.join(data_dir, 'b_least_3_data_cal.txt'))
	meas_data = b_read_meas_data(os.path.join(data_dir, 'b_least_3_data_meas.txt'))
	print('Linear function\n')
	b_test(cal_data, meas_data, b_linear_func)
	print('Power function\n')
	b_test(cal_data, meas_data, b_power_func)
	print('Exponential function\n')
	b_test(cal_data, meas_data, b_exp_func)

if __name__ == "__main__":
	b_example_1()
	b_example_2()
	b_example_3()
