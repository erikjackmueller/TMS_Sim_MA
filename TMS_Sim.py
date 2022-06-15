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
path = "Sphere_642"
sizes = [642, 1280]


class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)
man = MyManager()

if __name__ == '__main__':
    # functions.plot_default()
    n = 100
    r_max = 0.9955
    r = np.linspace(0.01, r_max, n)
    theta = np.linspace(0, 2*np.pi, n)
    phi = (1/2)*np.pi
    r0 = 10*np.array([0, 0.1, 1])
    # r0 = np.array([0.7071067811865476, 0.7071067811865476, 0])
    m = np.array([0, 0.1, 1])

    start = time.time()
    time_0 = start
    res1 = functions.reciprocity_three_D(r, theta, r0_v=r0, m=m, phi=phi)
    end = time.time()
    print(f"{end - start:.2f}s receprocity")

    start = time.time()
    # tc, areas = functions.read_sphere_mesh_from_txt(sizes, path)
    tc, areas = functions.read_sphere_mesh_from_txt_locations_only(sizes, path)
    # functions.triangulateSphere(20)
    end = time.time()
    print(f"{end - start:.2f}s triangulation")

    start = time.time()
    Q, rs = functions.SCSM_tri_sphere(tc, areas, r0=r0, m=m, sig=1e-9)
    # Q, rs = functions.SCSM_Q_parallel(man, tc, areas, r0=r0, m=m)
    end = time.time()
    print(f"{end - start:.2f}s  Q calculation")

    start = time.time()

    res = functions.parallel_SCSM_E_sphere(man, Q, rs, r, theta, r0=r0, m=m, phi=phi)
    end = time.time()
    print(f"{(end - start)/60:.2f}minutes E calculation")
    time_last = end
    print(f"{(time_last - time_0) / 60:.2f}minutes complete simulation")

    res2 = res.copy()

    diff_to_imag = np.abs(res1 - res2)

    rerror_imag = np.linalg.norm(diff_to_imag) / np.linalg.norm(res1)

    print("relative error:")
    print(rerror_imag)
    # print(res1[0, 0])
    # functions.plot_E(res1, r, theta, r_max)
    # functions.plot_E(res2, r, theta, r_max)
    # functions.plot_E(res2, r, theta, r_max)
    # functions.plot_E(diff_to_imag, r, theta, r_max)
    functions.plot_E_diff(res1, res2, r, theta, r_max)
    # functions.plot_E(res4, r, theta, r_max)
    # functions.plot_E(res1, r, theta, r_max)
    # functions.plot_E(diff, r, theta, r_max)



