import functions
import numpy as np
import time
import os
import matplotlib
matplotlib.use("TkAgg")
import multiprocessing.managers
from pathlib import Path
from functions import*

scaling_factor = 100
direction = np.array([1, 0, 1])
d_norm = direction/np.linalg.norm(direction)
r0 = 1.05 * d_norm * scaling_factor
m = d_norm
# path_string_original = "C:\Users\ermu8317\Downloads" #cannot be handled by python
# path_string = path_string_original.replace("\\", "/")
path = os.path.realpath(Path("C:/Users/ermu8317/Downloads"))
# path = os.path.realpath(Path("C:/Users/User/Downloads"))
fn = os.path.join(path, "15484.08.hdf5")
fn2 = os.path.join(path, "e.hdf5")
fn3 = "MagVenture_MCF_B65_REF_highres.ccd"


start = time.time()
time_0 = start
tc, areas, tri_points, n_v, tissue_types = read_mesh_from_hdf5(fn) #currently only csf points
r_targets = read_mesh_from_hdf5(fn, mode="target")
transformation_matrix, sigmas = read_mesh_from_hdf5(fn2, mode="coil")
print(f"# of elements: {tc.shape[0]}")
sigma = np.zeros(tc.shape[0])
tissue_number = tissue_types[-1] - 1000
for i in range(tissue_number):
    sigma[np.where(tissue_types == tissue_number + 1000)] = sigmas[i]
m, m_pos = read_from_ccd(path)
m_pos1 = np.vstack((m_pos.T, np.ones(m_pos.shape[0]))).T
m_pos1 = transformation_matrix @ m_pos1.T
m_pos = m_pos1[:3].T
m = 1e-5*np.array([1, 1, 0])

# m_pos.tofile('data.csv', sep = ',')
end = time.time()
# test in paraview tomorrow
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " triangulation")

start = time.time()
# Q, rs = functions.SCSM_tri_sphere(tc, areas, r0=r0, m=m, sig=1)
b_im = jacobi_vectors_numpy(tc, n_v, r0, m)
Q = SCSM_jacobi_iter_cupy(tc, areas, n_v, b_im, tol=1e-10, n_iter=20)
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " Q calculation")

start = time.time()
res = SCSM_FMM_E(Q=Q, r_source=tc, r_target=r_targets, eps=1e-15, m=m, r0=r0)
with h5py.File('test_e_field_csf.hdf5', 'w') as f:
    f.create_dataset('e', data=res)
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " E calculation")
time_last = end
t = t_format(time_last - time_0)
print(f"{t[0]:.2f}" + t[1] + " complete simulation")
