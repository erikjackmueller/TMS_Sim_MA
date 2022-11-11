from functions import*
import numpy as np
import time
import matplotlib
matplotlib.use("TkAgg")
from pathlib import Path
from matplotlib import cm

n = 100
phi1 = np.linspace(0, np.pi, n)
theta1 = np.linspace(0, 2 * np.pi, n)
phi2, theta2 = np.meshgrid(phi1, theta1)
phi, theta = phi2.T, theta2.T
scaling_factor = 1
r = 0.9
xyz_grid = xyz_grid(r, phi, theta)

direction = np.array([0, 0, 1])
d_norm = direction/np.linalg.norm(direction)
r0 = 1.05 * d_norm
# m, m_pos, transformation_matrix, sigmas = read_mesh_from_hdf5(fn2, mode="coil")
# m = d_norm
m = np.array([0, 0, 1])
omega = 3e3
# omega = 18.85956e3
# omega = 100
r_target = sphere_to_carthesian(r=r, phi=phi.flatten(), theta=theta.flatten())
# r_target = r_target[np.where(r_target[:, 2] > (0.8*r))]

start = time.time()
time_0 = start
# calculate analytic solution
res1 = reciprocity_sphere(grid=xyz_grid, r0_v=r0, m=m, omega=omega)[1]
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " receprocity")

start = time.time()
# tc, areas = functions.read_sphere_mesh_from_txt(sizes, path)
# tc, areas, tri_points = read_sphere_mesh_from_txt_locations_only(sizes, path, scaling=scaling_factor)
tc, areas, tri_points, n_v, avg_len = sphere_mesh(10000, scaling=scaling_factor)
print(f"average length: {avg_len}")
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " triangulation")
n_elements = tc.shape[0]
print(f"elements: {n_elements}")

start = time.time()
b_im = jacobi_vectors_numpy(tc, n_v, r0, m, omega=omega)
Q = SCSM_jacobi_iter_cupy(tc, areas, n_v, b_im, tol=5e-16, n_iter=20, omega=omega)
# Q = SCSM_jacobi_iter_numpy(tc, areas, n_v, b_im, tol=5e-16, n_iter=20, omega=omega)
# Q = SCSM_matrix(tc, areas, n=n_v, b_im=b_im, omega=omega)
# Q = SCSM_tri_sphere_numba(tc, tri_points, areas, r0=r0, m=m, omega=omega)[0]

rs = tc
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " Q jacobi")
b_im_ = vector_potential_for_E_single_m(rs=r_target, m=m, m_pos=r0, omega=omega)
start = time.time()


res_flat = SCSM_FMM_E2(Q=Q, r_source=tc, r_target=r_target, eps=1e-15, b_im=b_im_)
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
max_analytic = np.max(res1)
min_analytic = np.min(res1)
max_numeric = np.max(res)
min_numeric = np.min(res)

print(f"max_analytic = {max_analytic}, min_analytic = {min_analytic}, max_numeric = {max_numeric}, min_numeric = {min_numeric}")

# diff2 = np.abs(res1 - res3)
# relative_diff2 = diff2 / np.linalg.norm(res1)
#
error_imag = nrmse(res1, res2) * 100
# rerror_imag2 = np.linalg.norm(diff2) * 100 / np.linalg.norm(res1)

print(f"relative error: {error_imag:.7f}%")
# print(f"relative error with near field: {rerror_imag2:.7f}%")

# plot_E_sphere_surf(res, phi, theta, r)
plot_E_sphere_surf_diff(res1, res2, xyz_grid=xyz_grid, c_map=cm.jet)
# functions.plot_E_sphere_surf(res2, phi, theta, r)
# functions.plot_E_sphere_surf(diff, phi, theta, r)
# functions.plot_E_sphere_surf(relative_diff, phi, theta, r)






