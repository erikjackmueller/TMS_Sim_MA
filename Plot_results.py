from functions import*
import numpy as np
import time
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import cm
import matplotlib.pyplot as plt


# fig = plt.figure(figsize=(600, 1000))
fig, axs = plt.subplots(figsize=(12, 14), nrows=2, ncols=2)
methods = ["jacobi", "matrix"]
orientations = ["radial", "tangential"]
pythonlabels = True
yticks = np.arange(15, 0.0, -1.0)

#
value_under_test = 'r_in'
location = "results/r_in/"
labels = []
maxs = []

for orientation in orientations:
    for method in methods:
        values = np.loadtxt(location + "-" + method + "-" + orientation + "-" + value_under_test + "_test_values.csv",
                       delimiter=",")
        errors = np.loadtxt(location + "-" + method + "-" + orientation + "-" + value_under_test + "_test_errors.csv",
                            delimiter=",")
        maxs.append(errors.max())
        axs[0, 0].plot(values, errors, 'x--', alpha=0.9)
        labels.append(method + ", " + orientation)
y_max = np.max(np.array(maxs))
x_max = np.max(values)
x_min = np.min(values)

axs[0, 0].legend(labels)
axs[0, 0].set_yticks(yticks)
axs[0, 0].set_xticks(values)
if pythonlabels:
    axs[0, 0].set_ylabel("nrmse")
axs[0, 0].set_ylim(0, y_max + 1)
axs[0, 0].set_xlim(x_min - 5e-3, x_max + 5e-3)
if pythonlabels:
    axs[0, 0].set_xlabel("r_in")
axs[0, 0].grid()
axs[0, 0].axhline(y=1, color='r', alpha=0.3)

value_under_test = 'r_out'
location = "results/r_out_more/"
labels = []
maxs = []

for orientation in orientations:
    for method in methods:
        values = np.loadtxt(location + "-" + method + "-" + orientation + "-" + value_under_test + "_test_values.csv",
                       delimiter=",")
        errors = np.loadtxt(location + "-" + method + "-" + orientation + "-" + value_under_test + "_test_errors.csv",
                            delimiter=",")
        maxs.append(errors.max())
        axs[0, 1].plot(values, errors, 'x--', alpha=0.9)
        labels.append(method + ", " + orientation)
y_max = np.max(np.array(maxs))
x_max = np.max(values)
x_min = np.min(values)

axs[0, 1].legend(labels)
axs[0, 1].set_yticks(yticks)
axs[0, 1].set_xticks(values)
if pythonlabels:
    axs[0, 1].set_ylabel("nrmse")
axs[0, 1].set_ylim(0, y_max + 2)
axs[0, 1].set_xlim(x_min - 2e-3, x_max + 2e-3)
if pythonlabels:
    axs[0, 1].set_xlabel("r_out")
axs[0, 1].grid()
axs[0, 1].axhline(y=1, color='r', alpha=0.3)

value_under_test = 'f'
location = "results/f/"
labels = []
maxs = []

for orientation in orientations:
    for method in methods:
        values = np.loadtxt(location + "-" + method + "-" + orientation + "-" + value_under_test + "_test_values.csv",
                       delimiter=",")
        errors = np.loadtxt(location + "-" + method + "-" + orientation + "-" + value_under_test + "_test_errors.csv",
                            delimiter=",")
        maxs.append(errors.max())
        axs[1, 0].plot(values, errors, 'x--', alpha=0.9)
        labels.append(method + ", " + orientation)
y_max = np.max(np.array(maxs))
x_max = np.max(values)
x_min = np.min(values)

axs[1, 0].legend(labels)
axs[1, 0].set_yticks(yticks)
axs[1, 0].set_xticks(values)
if pythonlabels:
    axs[1, 0].set_ylabel("nrmse")
axs[1, 0].set_ylim(0, y_max + 2)
axs[1, 0].set_xscale('log')
axs[1, 0].set_xlim(x_min - 0.15, x_max + 1.5e3)
if pythonlabels:
    axs[1, 0].set_xlabel("f")
# axs[1, 0].grid(True, which="both")
axs[1, 0].grid()
axs[1, 0].axhline(y=1, color='r', alpha=0.3)

value_under_test = 'elements'
location = "results/n_elements/"
labels = []
maxs = []

for orientation in orientations:
    for method in methods:
        values = np.loadtxt(location + "-" + method + "-" + orientation + "-" + value_under_test + "_test_values.csv",
                       delimiter=",")
        errors = np.loadtxt(location + "-" + method + "-" + orientation + "-" + value_under_test + "_test_errors.csv",
                            delimiter=",")

        maxs.append(errors.max())
        axs[1, 1].plot(values, errors, 'x--', alpha=0.9)
        labels.append(method + ", " + orientation)
        if method == 'jacobi':
            val_jac = values

y_max = np.max(np.array(maxs))
x_max = np.max(val_jac)
x_min = np.min(val_jac)
axs[1, 1].legend(labels)
axs[1, 1].set_yticks(yticks)
axs[1, 1].set_xticks(values)
if pythonlabels:
    axs[1, 1].set_ylabel("nrmse")
axs[1, 1].set_ylim(0, y_max + 2)
axs[1, 1].set_xscale('log')
# xticks_1 = np.array([100, 300, 600, 1000, 2000, 4000, 8000, 10000, 20000, 30000, 40000])
# xticks = np.sort(np.concatenate((val_jac, xticks_1)))
# axs[1, 1].set_xticks(xticks)
# x_labels = [str(np.format_float_scientific(x, 3)) for x in xticks]
# for idx in [1, 3, 5, 7, 9, 11, 14, 15, 17, 18, 20, 21]:
#     x_labels[idx] = ""
# x_labels = [x.replace('.e+0', '*10^') for x in x_labels]
# axs[1, 1].set_xticklabels(x_labels)
# plt.setp(axs[1, 1].get_xticklabels(), rotation=45, horizontalalignment='right')
axs[1, 1].set_xlim(1e2, x_max + 4e3)
if pythonlabels:
    axs[1, 1].set_xlabel("number of elements")
axs[1, 1].grid(True, which="both")
axs[1, 1].axhline(y=1, color='r', alpha=0.3)
plt.show()