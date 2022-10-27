from functions import*
import numpy as np
import time
import matplotlib
import datetime
from pathlib import Path
import os
matplotlib.use("TkAgg")

start = time.time()
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
n = 100
scaling_factor = 1
r = 0.85 * scaling_factor
phi1 = np.linspace(0, np.pi, n)
theta1 = np.linspace(0, 2 * np.pi, n)
phi2, theta2 = np.meshgrid(phi1, theta1)
phi, theta = phi2.T, theta2.T
direction = np.array([0, 0, 1])
d_norm = direction/np.linalg.norm(direction)
r0 = 1.05 * d_norm * scaling_factor
r_target = sphere_to_carthesian(r=r, phi=phi.flatten(), theta=theta.flatten())
m = np.array([0, 1, 0])
xyz_grid = xyz_grid(r, phi, theta)
# calculate analytic solution
res1 = reciprocity_sphere(grid=xyz_grid, r0_v=r0, m=m, omega=3e3)

# create concentric sphere meshes with different radii
# radii = np.array([0.8, 0.9, 0.905, 0.91, 1.0])
radii = np.array([0.2, 0.4, 0.6, 0.7, 1.0])
sigmas = np.array([0.126, 0.275, 1.654, 0.001, 0.456])
# sigmas = np.array([0.126, 0.275, 1.654, 1.0, 0.456])
# sigmas = 0.33 * np.array([1, 1.01, 1.02, 1.03, 1.04, 1.05])
tc, areas, tri_points, n_v, sigmas_in, sigmas_out = layered_sphere_mesh(n_samples=1990, sigmas=sigmas, radii=radii)
print(f"elements: {tc.shape[0]}")
n_elem = tc.shape[0]
rs = tc

# b_im = jacobi_vectors_cupy(rs=tc, n=n_v, m=m, m_pos=m_pos, omega=3e3)
b_im = jacobi_vectors_numpy(rs, n=n_v, m=m, omega=3e3, r0=r0)
Q = SCSM_jacobi_iter_cupy(tc, areas, n_v, b_im, sig_in=sigmas_in, sig_out=sigmas_out, tol=1e-18, n_iter=2,
                         omega=3e3)
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " jacobi  Q")
# start = time.time()
# Q1 = SCSM_matrix(tc, areas, n=n_v, sig_in=sigmas_in, sig_out=sigmas_out, b_im=b_im, omega=3e3)
# end = time.time()
# t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " matrix  Q")
start = time.time()

# print(f" relative difference= {np.abs((np.linalg.norm(Q.imag) - np.linalg.norm(Q1.imag)) / (np.linalg.norm(Q1.imag) * 100)):.5f}%")

# b_im_ = vector_potential_for_E(r_target, m=m, m_pos=m_pos, omega=3e3)
b_im_ = vector_potential_for_E_single_m(rs=r_target, m=m, m_pos=r0, omega=3e3)

res_flat = SCSM_FMM_E2(Q=Q, r_source=tc, r_target=r_target, eps=1e-15, b_im=b_im_)
res = array_unflatten(res_flat, n_rows=n)
# res_flat = SCSM_FMM_E(Q=Q1, r_source=rs, r_target=r_target, eps=1e-20, r0=r0, omega=3e3)
# res2 = array_unflatten(res_flat, n_rows=n)
print(f"numeric E[0,0] {res[0, 0]}")
print(f"analytic E[0,0] {res1[0, 0]}")
plot_E_sphere_surf_diff(res1, res, phi, theta, r, names=["analytic", "jacobi"], c_map=cm.jet)
# plot_E_sphere_surf_diff(res1, res2, phi, theta, r, names=["analytic", "matrix"], c_map=cm.jet)
# plot_E_sphere_surf_diff(res, res2, phi, theta, r, normalize=True, names=["jacobi", "matrix"], c_map=cm.jet)

# plot_E_sphere_surf(res, phi, theta, r)
print("---------------------------")







