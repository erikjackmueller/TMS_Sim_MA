from functions import*
import numpy as np
import time
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import cm
import os
import tracemalloc

n = 400
r_out = 1.05
r_in = 0.9
scaling_factor = 1

# define dipole
r0_direction = np.array([0, 0, 1])
m = np.array([0, 0, 1])
omega = 18.85956e3

methods = ["jacobi", "matrix"]
orientations = ["radial", "tangential"]

# test r_in jacobi
# average length: 0.9188807652297692
# elements: 3983
value_under_test = 'elements'
samples = [150, 300, 500, 1000, 2000, 4000, 8000, 10000, 12000, 15000, 17000, 20000]
count = 0
location = "results/n_elements/"

# option to overwrite existing files
overwrite = True

for orientation in orientations:
    if orientation == "radial":
        m = np.array([0, 0, 1])
    elif orientation == "tangential":
        m = np.array([1, 0, 0])
    for method in methods:
        errors = []
        values = []
        for i in range(len(samples)):
            start = time.time()
            fn_fig = location + "-" + method + "-" + orientation + "-" + value_under_test + \
                     "-" + str(np.round(samples[i], 2)) + ".png"
            if not os.path.isfile(fn_fig) or overwrite:
                res1, res, xyz_grid, n_elements = One_layer_sphere_single_m_test(n=n, r_out=r_out, r_in=r_in,
                                                                     direction=r0_direction, m=m, omega=omega,
                                                                     n_samples=samples[i], method=method, tol=5e-16,
                                                                     n_iter=20, scaling_factor=scaling_factor,
                                                                     print_time=False, return_elements_nr=True)

                errors.append(np.round(nrmse(res, res1) * 100, 2))
                values.append(np.round(n_elements, 2))
                plot_E_sphere_surf_diff(res1, res, xyz_grid=xyz_grid, c_map=cm.jet,
                                            plot_difference=False, save=True, save_fn=fn_fig)
            count += 1
            print("----------------------------------------------------------")
            end = time.time()
            time_needed = end-start
            t = t_format(end - start)
            iter_left = (2 * 2 * len(samples)) - count
            t_left = t_format(time_needed * iter_left)
            print(fn_fig[:-4])
            print(f"iteration {count}/{2 * 2 * len(samples)} done!")
            print(f"iteration time needed: {t[0]:.2f}" + t[1])
            print(f"time left approximately: {t_left[0]:.2f}" + t_left[1])
            print("----------------------------------------------------------")

        fn_error = location + "-" + method + "-" + orientation + "-" + value_under_test + "_test_errors.csv"
        if not os.path.isfile(fn_error) or overwrite:
            np.savetxt(fn_error, np.array(errors), delimiter=",")
        fn_values = location + "-" + method + "-" + orientation + "-" + value_under_test + "_test_values.csv"
        if not os.path.isfile(fn_values) or overwrite:
            np.savetxt(fn_values, np.array(values), delimiter=",")


# tracemalloc.start()
#
# # function call
#
# # displaying the memory
# print(tracemalloc.get_traced_memory())



