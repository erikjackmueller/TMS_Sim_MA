from functions import*
import numpy as np
import time
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import cm
import matplotlib.pyplot as plt


small = True
fig, axs = plt.subplots(figsize=(10, 6), nrows=2, ncols=1)
elements = np.array([487, 983, 1484, 1981, 2984, 3983])
memory = np.loadtxt("Q_benchmark_memory", delimiter=",")
#fix cupy JIT
memory[4, 0] = 1/8 * memory[4, 0]
time = np.loadtxt("Q_benchmark_time", delimiter=",")
elements = np.loadtxt("Q_benchmark_elements", delimiter=",")
memory_large = np.loadtxt("Q_benchmark_large_samples_memory", delimiter=",")
time_large = np.loadtxt("Q_benchmark_large_samples_time", delimiter=",")
elements_large = np.loadtxt("Q_benchmark_large_samples_elements", delimiter=",")
width = 50
width2 = 4000
axs[0].set_ylabel("peak memory usage in MB")
axs[1].set_ylabel("t in s")
y_ticks = [1, 10, 60, 600, 3600, 5*3600]
y_labels = ["1s", "10s", "1 min", "10 min", "1h", "5h"]
if small:
        axs[0].set_ylim([1e-2, 1e4])
        axs[1].set_ylim([1e-1, 2e3])
        for i in range(5):
                axs[0].bar(elements - 115 + ((15 +width)*i), memory[i], width=width, edgecolor='black')
                axs[1].bar(elements - 115 + ((15 +width)*i), time[i], width=width, edgecolor='black')

else:
        axs[0].set_ylim([1e-1, 3e2])
        axs[1].set_ylim([1, 1e5])

                # axs[0].bar(elements_large[:4] - 200 + ((500 + width2) * i), memory_large[i][:4], width=np.diff(memory_large[i][:5])*10, edgecolor='black')
                # axs[1].bar(elements_large[:4] - 200 + ((500 + width2) * i), time_large[i][:4], width=np.diff(time_large[i][:5])*10, edgecolor='black')
        axs[0].plot(elements_large[:5], memory_large[0][:5], "x--", alpha=0.9, lw=2, color='red')
        axs[1].plot(elements_large[:5], time_large[0][:5], "x--", alpha=0.9, lw=2, color='red')
        axs[0].plot(elements_large[:5], memory_large[1][:5], "x--", alpha=0.9, lw=2, color='purple')
        axs[1].plot(elements_large[:5], time_large[1][:5], "x--", alpha=0.9, lw=2, color='purple')
if small:
        for j in range(2):
                axs[j].set_yscale("log")
                axs[j].set_axisbelow(True)
                axs[j].grid(True, which='both', axis='y', zorder=-25.0)
                axs[j].set_xlabel("number of elements")
                axs[j].legend(["python loop", "numba", "jacobi", "vectorized jacobi", "vectorized jacobi (cupy)"],
                              loc="upper left", fontsize=7)
        axs[0].axhline(y=1, color='k', alpha=0.3)
        axs[0].axhline(y=1e1, color='k', alpha=0.3)
        axs[0].axhline(y=1e2, color='k', alpha=0.3)
        axs[0].axhline(y=1e3, color='k', alpha=0.3)
        for e in y_ticks[:-1]:
                axs[1].axhline(y=e, color='k', alpha=0.3)
        axs[1].set_yticks(y_ticks[:-1])
        axs[1].set_yticklabels(y_labels[:-1])

else:
        for j in range(2):
                axs[j].set_xscale("log")
                axs[j].set_yscale("log")
                axs[j].set_axisbelow(True)
                axs[j].grid(True, which='both', axis='y', zorder=-25.0)
                axs[j].set_xticks([x for x in elements_large[:5]])
                axs[j].set_xticklabels([str(np.round(x, -3))[:-2] for x in elements_large[:5]])
                axs[j].set_xlim(3900, 210000)
                axs[j].set_xlabel("number of elements")
                axs[j].legend(["vectorized jacobi", "vectorized jacobi (cupy)"],
                              loc="upper left", fontsize=7)

        axs[0].axhline(y=1, color='k', alpha=0.3)
        axs[0].axhline(y=1e1, color='k', alpha=0.3)
        axs[0].axhline(y=1e2, color='k', alpha=0.3)
        axs[0].axhline(y=1e3, color='k', alpha=0.3)
        axs[0].axhline(y=1e4, color='k', alpha=0.3)

        for e in y_ticks:
                axs[1].axhline(y=e, color='k', alpha=0.3)
        axs[1].set_yticks(y_ticks)
        axs[1].set_yticklabels(y_labels)

fig.tight_layout()

plt.show()




# width = 0.25
#
# plt.bar(r, Women, color='b',
#         width=width, edgecolor='black',
#         label='Women')
# plt.bar(r + width, Men, color='g',
#         width=width, edgecolor='black',
#         label='Men')
#
# plt.xlabel("Year")
# plt.ylabel("Number of people voted")
# plt.title("Number of people voted in each year")
#
# # plt.grid(linestyle='--')
# plt.xticks(r + width / 2, ['2018', '2019', '2020', '2021'])
# plt.legend()
#
# plt.show()