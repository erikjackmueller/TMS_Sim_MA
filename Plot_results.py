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
yticks = np.arange(10, 0.0, -1.0)

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
location = "results/r_out/"
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
axs[1, 0].grid()
axs[1, 0].axhline(y=1, color='r', alpha=0.3)
plt.show()

value_under_test = 'elements'
location = "results/n_elements/"
labels = []
maxs = []