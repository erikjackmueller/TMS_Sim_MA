import functions
import numpy as np
import os
import time
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import multiprocessing.managers



# functions.plot_default()
# path = os.path.realpath(Path("C:/Users/Besitzer/Downloads/Sphere_642"))
path = "Sphere_624"
sizes = [624, 1280]


class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)
man = MyManager()

if __name__ == '__main__':
    # functions.plot_default()
    n = 100
    r_max = 1
    r = np.linspace(0.01, r_max, n)
    theta = np.linspace(0, 2*np.pi, n)
    phi = (1/2)*np.pi
    r0 = np.array([3, 0, 0])
    m = np.array([0, -1, 0])

    start = time.time()
    time_0 = start
    res1 = functions.reciprocity_three_D(r, theta, r0_v=r0, m=m, phi=phi)
    end = time.time()
    print(f"{end - start:.2f}s receprocity")

    start = time.time()
    tc, areas = functions.read_sphere_mesh_from_txt_locations_only(sizes, path)
    # functions.triangulateSphere(20)
    end = time.time()
    print(f"{end - start:.2f}s triangulation")

    start = time.time()
    # Q, rs = functions.SCSM_tri_sphere(tc, areas, r0=r0, m=m)
    Q, rs = functions.SCSM_Q_parallel(man, tc, areas, r0=r0, m=m)
    end = time.time()
    print(f"{end - start:.2f}s  Q calculation")

    start = time.time()

    res = functions.parallel_SCSM_E_sphere(man, Q, rs, r, theta, r0=r0, m=m, phi=phi)
    end = time.time()
    print(f"{(end - start)/60:.2f}minutes E calculation")
    time_last = end
    print(f"{(time_last - time_0) / 60:.2f}minutes complete simulation")

    res2 = res[1]
    res3 = res[2]
    res4 = res[3]

    # diff_to_real = res1 - res2
    diff_to_imag = res1 - res3
    # diff_to_comp = res1 - res4

    # rerror_real = np.linalg.norm(diff_to_real) / np.linalg.norm(res1)
    rerror_imag = np.linalg.norm(diff_to_imag) / np.linalg.norm(res1)
    # rerror_comp = np.linalg.norm(diff_to_comp) / np.linalg.norm(res1)

    print("relative error:")
    # print(rerror_real)
    print(rerror_imag)
    # print(rerror_comp)

    # rel_error = np.linalg.norm(res1 - res) / np.linalg.norm(res1)
    # print(f"relative error = {rel_error}")
    # diff = res - res1
    functions.plot_E(res1, r, theta, r_max)
    # functions.plot_E(res2, r, theta, r_max)
    functions.plot_E(res3, r, theta, r_max)
    functions.plot_E(diff_to_imag, r, theta, r_max)
    # functions.plot_E(res4, r, theta, r_max)
    # functions.plot_E(res1, r, theta, r_max)
    # functions.plot_E(diff, r, theta, r_max)



