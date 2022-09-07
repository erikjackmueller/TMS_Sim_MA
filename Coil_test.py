from functions import*
import numpy as np
import time
import matplotlib
import datetime
from pathlib import Path
import os
matplotlib.use("TkAgg")

path = os.path.realpath(Path("C:/Users/ermu8317/Downloads"))
# path = os.path.realpath(Path("C:/Users/User/Downloads"))
fn = os.path.join(path, "15484.08.hdf5")
fn2 = os.path.join(path, "e.hdf5")
fn3 = "MagVenture_MCF_B65_REF_highres.ccd"

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
transformation_matrix, sigmas = read_mesh_from_hdf5(fn2, mode="coil")

x_change = 1.05
transformation_matrix = np.eye(4)
transformation_matrix[3, 0] = x_change


m, m_pos = read_from_ccd(path)
m_pos = m_pos * 1e-5

m_pos1 = np.vstack((m_pos.T, np.ones(m_pos.shape[0]))).T
m_pos1 = transformation_matrix @ m_pos1.T
m_pos = m_pos1[:3].T
m_pos.tofile('data.csv', sep = ',')
end = time.time()
r_target = sphere_to_carthesian(r=r, phi=phi.flatten(), theta=theta.flatten())
r0 = m_pos

tc, areas, tri_points, n_v = sphere_mesh(1000, scaling=scaling_factor)[:4]
simgas_test = np.random.rand(tc.shape[0])

ig = False
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print(f"elements: {tc.shape[0]}")
n_elem = tc.shape[0]
rs = tc
start = time.time()
start_sub = start
b_im = jacobi_vectors_numpy(tc, n_v, m=m, m_pos=m_pos)
end_sub = time.time()
t_sub = t_format(end_sub - start_sub)
print(f"{t_sub[0]:.2f}" + t_sub[1] + "  b calculation")
Q = SCSM_jacobi_iter_cupy(tc, areas, n_v, b_im, sig=simgas_test, tol=1e-10, n_iter=20)
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + "  Q jacobi")
start = time.time()

b_im_ = vector_potential_for_E(r_target, m=m, m_pos=m_pos)
res_flat = SCSM_FMM_E2(Q=Q, r_source=rs, r_target=r_target, eps=1e-2, b_im=b_im_)
res = array_unflatten(res_flat, n_rows=n)
plot_E_sphere_surf(res, phi, theta, r)
print("---------------------------")







