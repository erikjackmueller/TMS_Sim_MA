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
    n = 400
    r_max = 1
    r = np.linspace(0.01, r_max, n)
    theta = np.linspace(0, 2*np.pi, n)
    phi = (1/2)*np.pi
    r0 = np.array([1.2, 0, 0])
    m = np.array([0, 1, 0])

    start = time.time()
    res1 = functions.reciprocity_three_D(r, theta, r0_v=r0, m=m, phi=phi)
    end = time.time()
    print(f"{end - start:.2f}s receprocity")

    start = time.time()
    tc, areas = functions.read_sphere_mesh_from_txt_locations_only(sizes, path)
    # functions.triangulateSphere(20)
    end = time.time()
    print(f"{end - start:.2f}s triangulation")

    start = time.time()
    Q, rs = functions.SCSM_tri_sphere(tc, areas, r0=r0, m=m)
    end = time.time()
    print(f"{end - start:.2f}s  Q calculation")

    start = time.time()

    res = functions.parallel_SCSM_E_sphere(man, Q, rs, r, theta, r0=r0, m=m, phi=phi)
    end = time.time()
    print(f"{(end - start)/60:.2f}minutes E calculation")

    rel_error = np.linalg.norm(res1 - res) / np.linalg.norm(res1)
    print(f"relative error = {rel_error}")
    diff = res - res1
    # functions.plot_E(res, r, theta, r_max)
    # functions.plot_E(res1, r, theta, r_max)
    functions.plot_E(diff, r, theta, r_max)



