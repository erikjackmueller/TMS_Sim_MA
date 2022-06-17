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
path = "Sphere_10242"
sizes = [10242, 1280]
# path = "Sphere_642"
# sizes = [642, 1280]

class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)
man = MyManager()

r0_list = [np.array([0, 0, 1]), np.array([0, 0, 2]), np.array([0, 0, 10]),
           np.array([0, 1, 0]), np.array([0, 2, 0]), np.array([0, 10, 0]),
           np.array([1, 0, 0]), np.array([2, 0, 0]), np.array([10, 0, 0])]
r0_str_list = ["1z", "2z", "10z", "1y", "2y", "10y", "1x", "2x", "10x"]
m_list = [np.array([0, 0, -1]), np.array([0, -1, 0]), np.array([-1, 0, 0])]
m_str_list = ["z", "y", "x"]

if __name__ == '__main__':
    man.start()
    for i in range(9):
        for j in [0]:
            name = "r0_" + r0_str_list[i] + "_m_" + m_str_list[j]

            # functions.plot_default()
            n = 100
            r_max = 0.9955
            r = np.linspace(0.01, r_max, n)
            theta = np.linspace(0, 2*np.pi, n)
            phi = (1/2)*np.pi
            r0 = r0_list[i]
            # r0 = np.array([0.7071067811865476, 0.7071067811865476, 0])
            m = m_list[j]

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
            Q, rs = functions.SCSM_tri_sphere(tc, areas, r0=r0, m=m, sig=1)
            # Q, rs = functions.SCSM_Q_parallel(man, tc, areas, r0=r0, m=m)
            end = time.time()
            print(f"{end - start:.2f}s  Q calculation")

            start = time.time()

            res = functions.parallel_SCSM_E_sphere(man, Q, rs, r, theta, r0=r0, m=m, phi=phi)
            # res = functions.SCSM_E_sphere(Q, rs, r, theta, r0=r0, m=m)
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
            # functions.plot_E_diff(res1, res2, r, theta, r_max, r0, m)
            diff = np.abs(res2 - res1)
            fig, ax = plt.subplots(1, 4, subplot_kw={'projection': 'polar'}, figsize=(18, 4))
            ax0, ax1, ax2, ax3 = ax[0], ax[1], ax[2], ax[3]
            im1 = ax0.pcolormesh(theta, r, res1, cmap='plasma', vmin=res1.min(), vmax=res1.max())
            fig.colorbar(im1, ax=ax0)
            ax0.set_yticklabels([])
            ax0.set_rmax(r_max)
            ax0.grid(True)
            f_max = max(res1.max(), res2.max())
            f_min = min(res1.min(), res2.min())
            im = ax1.pcolormesh(theta, r, res1, cmap='plasma', vmin=f_min, vmax=f_max)
            ax1.set_yticklabels([])
            ax1.set_rmax(r_max)
            ax1.grid(True)
            im = ax2.pcolormesh(theta, r, res2, cmap='plasma', vmin=f_min, vmax=f_max)
            ax2.set_yticklabels([])
            ax2.set_rmax(r_max)
            ax2.grid(True)
            im = ax3.pcolormesh(theta, r, diff, cmap='plasma', vmin=f_min, vmax=f_max)
            ax3.set_yticklabels([])
            ax3.set_rmax(r_max)
            ax3.grid(True)
            fig.colorbar(im)
            ax0.set_title("analytic (original scale)")
            ax1.set_title("analytic (scaled to max-value)")
            ax2.set_title("numeric")
            ax3.set_title("difference")
            plt.subplots_adjust(wspace=0.7)
            rerror = np.linalg.norm(diff) / np.linalg.norm(res1)
            fig.suptitle(f"relative error: {rerror:.6f}, r0 = {r0}, m = {m}")
            plt.savefig(name + ".png")
            plt.close()
            # functions.plot_E_diff(res2, res3, r, theta, r_max, r0, m)
            # functions.plot_E(res4, r, theta, r_max)
            # functions.plot_E(res1, r, theta, r_max)
            # functions.plot_E(diff, r, theta, r_max)