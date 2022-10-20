from functions import*
import numpy as np
import time
import matplotlib
import datetime
from pathlib import Path
import os
matplotlib.use("TkAgg")

file_path = os.path.realpath(Path("C:/Users/ermu8317/Downloads"))
# path = os.path.realpath(Path("C:/Users/User/Downloads"))
fn = os.path.join(file_path, "15484.08.hdf5")
fn2 = os.path.join(file_path, "e.hdf5")
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
m, m_pos = read_from_ccd(file_path)
m_pos = translate(m_pos, transformation_matrix)
# m = translate(m, transformation_matrix)
# np.savetxt("coil0.csv", m_pos0, delimiter=",")

np.savetxt("coil.csv", m_pos, delimiter=",")
end = time.time()
r_target = sphere_to_carthesian(r=r, phi=phi.flatten(), theta=theta.flatten())
# loc_mat = np.eye(4)
# loc_mat[3, 0] = 3 # how did translation work?
# loc_mat[3, 1] = 3
# loc_mat[3, 2] = 3
# loc_mat[3, 3] = 3
m_pos = m_pos/200 + 0.75
tc, areas, tri_points, n_v = sphere_mesh(1000, scaling=scaling_factor)[:4]


# set up realistic sigma values
div = int(np.floor(tc.shape[0]/5))
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


ig = False
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print(f"elements: {tc.shape[0]}")
n_elem = tc.shape[0]
rs = tc
start = time.time()
start_sub = start
b_im = jacobi_vectors_cupy(rs=tc, n=n_v, m=m, m_pos=m_pos, omega=3e3)
end_sub = time.time()
t_sub = t_format(end_sub - start_sub)
print(f"{t_sub[0]:.2f}" + t_sub[1] + "  b calculation")
Q = SCSM_jacobi_iter_cupy(tc, areas, n_v, b_im, sig_in=sigmas_in_test, sig_out=sigmas_out_test, tol=1e-15, n_iter=100,
                          omega=3e3)
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + "  Q jacobi")
start = time.time()

b_im_ = vector_potential_for_E(r_target, m=m, m_pos=m_pos, omega=3e3)
res_flat = SCSM_FMM_E2(Q=Q, r_source=rs, r_target=r_target, eps=1e-20, b_im=b_im_)
res = array_unflatten(res_flat, n_rows=n)
print(f"{res[0, 0]}")
plot_E_sphere_surf(res, phi, theta, r)
print("---------------------------")







