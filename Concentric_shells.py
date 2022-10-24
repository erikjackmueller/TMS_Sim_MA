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
direction = np.array([1, 0, 1])
d_norm = direction/np.linalg.norm(direction)
r0 = 1.05 * d_norm * scaling_factor
r_target = sphere_to_carthesian(r=r, phi=phi.flatten(), theta=theta.flatten())
m = d_norm

# create concentric sphere meshes with different radii
radii = np.array([0.8, 0.9, 0.905, 0.91, 1.0])
r_shells = []
for n in range(radii.shape[0]):
    tc_n, areas_n, tri_points_n, n_v_n = sphere_mesh(1000, scaling=radii[n])[:4]
    if n == 0:
        div = tc_n.shape[0]
        tc = tc_n
        areas = areas_n
        tri_points = tri_points_n
        n_v = n_v_n
    if n > 0:
        tc = np.vstack((tc, tc_n))
        areas = np.concatenate((areas, areas_n))
        tri_points = np.vstack((tri_points, tri_points_n))
        n_v = np.vstack((n_v, n_v_n))
# set up realistic sigma values
sigmas_in_test = np.zeros(tc.shape[0])
sigmas_in_test[:div] = 0.126
sigmas_in_test[div:(2*div)] = 0.275
sigmas_in_test[(2*div):(3*div)] = 1.654
sigmas_in_test[(3*div):(4*div)] = 0.001
sigmas_in_test[(4*div):(5*div + 1)] = 0.456
sigmas_out_test = np.zeros_like(sigmas_in_test)
sigmas_out_test[:div] = 0.275
sigmas_out_test[div:(2*div)] = 1.654
sigmas_out_test[(2*div):(3*div)] = 0.001
sigmas_out_test[(3*div):(4*div)] = 0.456
sigmas_out_test[(4*div):(5*div + 1)] = 0.000


print(f"elements: {tc.shape[0]}")
n_elem = tc.shape[0]
rs = tc
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + "  Q jacobi")
start = time.time()
start = time.time()
start_sub = start
# b_im = jacobi_vectors_cupy(rs=tc, n=n_v, m=m, m_pos=m_pos, omega=3e3)
b_im = jacobi_vectors_numpy(rs, n, m=m, omega=3e3, m_pos=r0)
end_sub = time.time()
t_sub = t_format(end_sub - start_sub)
print(f"{t_sub[0]:.2f}" + t_sub[1] + "  b calculation")
Q = SCSM_jacobi_iter_cupy(tc, areas, n_v, b_im, sig_in=sigmas_in_test, sig_out=sigmas_out_test, tol=1e-15, n_iter=100,
                          omega=3e3)
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + "  Q jacobi")
start = time.time()

# b_im_ = vector_potential_for_E(r_target, m=m, m_pos=m_pos, omega=3e3)
res_flat = SCSM_FMM_E(Q=Q, r_source=rs, r_target=r_target, eps=1e-20, omega=3e3)
res = array_unflatten(res_flat, n_rows=n)
print(f"{res[0, 0]}")
plot_E_sphere_surf(res, phi, theta, r)
print("---------------------------")







