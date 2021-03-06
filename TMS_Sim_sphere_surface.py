from functions import*
import numpy as np
import time
import matplotlib
matplotlib.use("TkAgg")
import multiprocessing.managers
from matplotlib import cm



# functions.plot_default()
# path = os.path.realpath(Path("C:/Users/Besitzer/Downloads/Sphere_642"))
# path = "Sphere_10242"
# sizes = [10242, 1280]
path = "Sphere_2964"
sizes = [2964, 1280]
# path = "Sphere_642"
# sizes = [642, 1280]

# class MyManager(multiprocessing.managers.BaseManager):
#     pass
# MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)
# man = MyManager()
# if __name__ == '__main__':
#     man.start()
n = 100
scaling_factor = 1
r = 0.97 * scaling_factor
phi1 = np.linspace(0, np.pi, n)
theta1 = np.linspace(0, 2 * np.pi, n)
phi2, theta2 = np.meshgrid(phi1, theta1)
phi, theta = phi2.T, theta2.T
direction = np.array([1, 0, 1])
d_norm = direction/np.linalg.norm(direction)
r0 = 1.05 * d_norm * scaling_factor
m = d_norm
r_target = sphere_to_carthesian(r=r, phi=phi.flatten(), theta=theta.flatten())



start = time.time()
time_0 = start
res1 = reciprocity_three_D(r, theta, r0_v=r0, m=m, phi=phi, projection="sphere_surface")
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " receprocity")

start = time.time()
# tc, areas = functions.read_sphere_mesh_from_txt(sizes, path)
# tc, areas, tri_points = read_sphere_mesh_from_txt_locations_only(sizes, path, scaling=scaling_factor)
tc, areas, tri_points, n_v, avg_len = sphere_mesh(1000, scaling=scaling_factor)
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " triangulation")
n_elements = tc.shape[0]
print(f"elements: {n_elements}")

# start = time.time()
# Q, rs = SCSM_tri_sphere_numba(tc, tri_points, areas, r0=r0, m=m)
# end = time.time()
# t = t_format(end - start)
# print(f"{t[0]:.2f}" + t[1] + "  Q linalg.solve()")

start = time.time()
Q = SCSM_jacobi_iter_debug(tc, tri_points, areas, n=n_v, r0=r0, m=m, tol=1e-2, initial_guess=False)
rs = tc
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + "  Q jacobi")

start = time.time()

res_flat = SCSM_FMM_E(Q=Q, r_source=rs, r_target=r_target, eps=1e-2, m=m, r0=r0)
res = array_unflatten(res_flat, n_rows=n)
# res3 = functions.parallel_SCSM_E_sphere(man, Q, rs, r, theta=theta, phi=phi, r0=r0, m=m, projection="sphere_surface")
# res = functions.SCSM_E_sphere(Q, rs, r, theta, r0=r0, m=m)
# res = SCSM_E_sphere_numba_surf(Q, rs, r, theta, r0=r0, m=m, phi=phi)

end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + "  E calculation")

# start = time.time()
# res3 = E_near_correction(E=res, Q=Q, r_q=rs, r_sphere=r, tri_points=tri_points, theta=theta, phi=phi, n=n_elements,
#                          r_near=3*avg_len)
# end = time.time()
# t = t_format(end - start)
# print(f"{t[0]:.2f}" + t[1] + "  E correction")

time_last = end
t = t_format(end - time_0)
print(f"{t[0]:.2f}" + t[1] + "  complete simulation")

res2 = res.copy()

diff = np.abs(res1 - res2)
relative_diff = diff / np.linalg.norm(res1)
# diff2 = np.abs(res1 - res3)
# relative_diff2 = diff2 / np.linalg.norm(res1)
#
rerror_imag = np.linalg.norm(diff) * 100 / np.linalg.norm(res1)
# rerror_imag2 = np.linalg.norm(diff2) * 100 / np.linalg.norm(res1)

print(f"relative error: {rerror_imag:.7f}%")
# print(f"relative error with near field: {rerror_imag2:.7f}%")

# functions.plot_E_sphere_surf(res, phi, theta, r)
# plot_E_sphere_surf_diff(res1, res2, phi, theta, r, c_map=cm.coolwarm)
# functions.plot_E_sphere_surf(res2, phi, theta, r)
# functions.plot_E_sphere_surf(diff, phi, theta, r)
# functions.plot_E_sphere_surf(relative_diff, phi, theta, r)






