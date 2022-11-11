import functions
import numpy as np
import time
import os
import matplotlib
matplotlib.use("TkAgg")
import multiprocessing.managers
from pathlib import Path
import datetime
from functions import*

# path_string_original = "C:\Users\ermu8317\Downloads" #cannot be handled by python
# path_string = path_string_original.replace("\\", "/")
path = os.path.realpath(Path("C:/Users/ermu8317/Downloads"))
# path = os.path.realpath(Path("C:/Users/User/Downloads"))
fn = os.path.join(path, "15484.08.hdf5")
fn2 = os.path.join(path, "e.hdf5")
fn3 = "MagVenture_MCF_B65_REF_highres.ccd"

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
start = time.time()
time_0 = start
tc, areas, tri_points, n_v, tissue_types = read_mesh_from_hdf5(fn)
r_targets = read_mesh_from_hdf5(fn, mode="target")
transformation_matrix, sigmas = read_mesh_from_hdf5(fn2, mode="coil")
print(f"# of elements: {tc.shape[0]}")
sigma_out = np.zeros(tc.shape[0])
sigma_in = np.zeros_like(sigma_out)
tissue_number = np.max(tissue_types) - 1000
for i in range(tissue_number + 1):
    sigma_in[np.where(tissue_types == i + 1000)] = sigmas[i - 1]
    sigma_out[np.where(tissue_types == i + 1000)] = sigmas[i]
m, m_pos_raw = read_from_ccd(path)
m_pos = translate_old(m_pos_raw, transformation_matrix)
omega = 18.85956e3
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " triangulation")

start = time.time()
# Q, rs = functions.SCSM_tri_sphere(tc, areas, r0=r0, m=m, sig=1)
b_im = jacobi_vectors_cupy(rs=tc, n=n_v, m=m, m_pos=m_pos, omega=omega)
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " b calculation")
start = time.time()
Q = SCSM_jacobi_iter_cupy(tc, areas, n_v, b_im, tol=1e-14, n_iter=10, omega=omega, sig_in=sigma_in, sig_out=sigma_out,
                          verbose=True)
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " Q calculation")
print(f"Q: {Q[0], Q[-1]}")

start = time.time()
b_im_ = vector_potential_for_E(rs=r_targets, m=m, m_pos=m_pos, omega=omega)
print(f"b for E: {b_im_[0], b_im_[-1]}")
# b_im_ = vector_potential_for_E(rs=r_targets, n=n_v, m=m, m_pos=m_pos)
res = SCSM_FMM_E2(Q=Q, r_source=tc, r_target=r_targets, eps=1e-15, b_im=b_im_)
print(f"E: {res[0], res[-1]}")
# res = SCSM_FMM_E(Q=Q, r_source=tc, r_target=r_targets, eps=1e-15, m=m, r0=r0)
with h5py.File('e_field_midlayer_m1s1_pmd_scsm.hdf5', 'w') as f:
    f.create_dataset('e', data=res)
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " E calculation")
time_last = end
t = t_format(time_last - time_0)
print(f"{t[0]:.2f}" + t[1] + " complete simulation")
