import functions
import numpy as np
import time
import matplotlib
matplotlib.use("TkAgg")
import multiprocessing.managers



# functions.plot_default()
# path = os.path.realpath(Path("C:/Users/Besitzer/Downloads/Sphere_642"))
# path = "Sphere_10242"
# sizes = [10242, 1280]
# path = "Sphere_2964"
# sizes = [2964, 1280]
path = "Sphere_642"
sizes = [642, 1280]

class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)
man = MyManager()
if __name__ == '__main__':
    # functions.plot_default()
    man.start()
    n = 100
    r_max = 0.9
    r = 0.8
    theta = np.linspace(0, 2*np.pi, n)
    phi = np.linspace(0, np.pi, n)
    # r0 = np.array([0, 1.05, 0])
    r0 = 1.05*np.array([1, 0, 0])
    m = np.array([-1, 0, 0])

    start = time.time()
    time_0 = start
    res1 = functions.reciprocity_three_D(r, theta, r0_v=r0, m=m, phi=phi, projection="sphere_surface")
    end = time.time()
    print(f"{end - start:.2f}s receprocity")

    start = time.time()
    # tc, areas = functions.read_sphere_mesh_from_txt(sizes, path)
    tc, areas = functions.read_sphere_mesh_from_txt_locations_only(sizes, path)
    # functions.triangulateSphere(20)
    end = time.time()
    print(f"{end - start:.2f}s triangulation")

    start = time.time()
    # Q, rs = functions.SCSM_tri_sphere(tc, areas, r0=r0, m=m, sig=1)
    Q, rs = functions.SCSM_tri_sphere_numba(tc, areas, r0=r0, m=m)
    end = time.time()
    print(f"{end - start:.2f}s  Q calculation")

    start = time.time()

    res = functions.parallel_SCSM_E_sphere(man, Q, rs, r, theta=theta, phi=phi, r0=r0, m=m, projection="sphere_surface")
    # res = functions.SCSM_E_sphere(Q, rs, r, theta, r0=r0, m=m)
    # res = functions.numba_SCSM_E_sphere(Q, rs, r, theta, r0=r0, m=m)
    end = time.time()
    print(f"{(end - start)/60:.2f}minutes E calculation")
    time_last = end
    print(f"{(time_last - time_0) / 60:.2f}minutes complete simulation")

    res2 = res.copy()

    diff = np.abs(res1 - res2)
    relative_diff = diff / np.linalg.norm(res1)
    #
    rerror_imag = np.linalg.norm(diff) / np.linalg.norm(res1)

    print("relative error:")
    print(rerror_imag)
    functions.plot_E_sphere_surf(res1, phi, theta, r)
    functions.plot_E_sphere_surf(res2, phi, theta, r)
    functions.plot_E_sphere_surf(diff, phi, theta, r)
    # functions.plot_E_sphere_surf(relative_diff, phi, theta, r)






