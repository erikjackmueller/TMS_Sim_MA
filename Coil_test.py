from functions import*
import numpy as np
import time
import matplotlib
import datetime
from pathlib import Path
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
m, m_pos = read_from_ccd(path)
m_pos1 = np.vstack((m_pos.T, np.ones(m_pos.shape[0]))).T
m_pos1 = transformation_matrix @ m_pos1.T
m_pos = m_pos1[:3].T
m_pos.tofile('data.csv', sep = ',')
end = time.time()
r_target = sphere_to_carthesian(r=r, phi=phi.flatten(), theta=theta.flatten())

tc, areas, tri_points, n_v = sphere_mesh(1000, scaling=scaling_factor)[:4]

ig = False
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print(f"elements: {tc.shape[0]}")
n_elem = tc.shape[0]
rs = tc
start = time.time()
start_sub = start
b_im = jacobi_vectors_numpy(tc, n_v, r0, m)
end_sub = time.time()
t_sub = t_format(end_sub - start_sub)
print(f"{t_sub[0]:.2f}" + t_sub[1] + "  b calculation")
Q = SCSM_jacobi_iter_cupy(tc, areas, n_v, b_im, tol=1e-10, n_iter=20)
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + "  Q jacobi")
start = time.time()

res_flat = SCSM_FMM_E(Q=Q, r_source=rs, r_target=r_target, eps=1e-2, m=m, r0=r0)
res = array_unflatten(res_flat, n_rows=n)
plot_E_sphere_surf(res, phi, theta, r)
print("---------------------------")







