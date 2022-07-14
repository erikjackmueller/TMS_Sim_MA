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
# path = os.path.realpath(Path("C:/Users/ermu8317/Downloads"))
path = os.path.realpath(Path("C:/Users/User/Downloads"))
fn = os.path.join(path, "15484.08.hdf5")

start = time.time()
time_0 = start
tc, areas, tri_points = read_mesh_from_hdf5(fn) #currently only csf points
r_targets = read_mesh_from_hdf5(fn, mode="target")
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + "triangulation")

start = time.time()
# Q, rs = functions.SCSM_tri_sphere(tc, areas, r0=r0, m=m, sig=1)
Q, rs = functions.SCSM_tri_sphere_dask(tc, tri_points, areas, r0=r0, m=m)
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + "Q calculation")

start = time.time()
res = SCSM_FMM_E(Q=Q, r_source=rs, r_target=r_targets, eps=1e-15, m=m, r0=r0)
with h5py.File('test_e_field_csf.hdf5', 'w') as f:
    f.create_dataset('e', data=res)
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " E calculation")
time_last = end
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " complete simulation")
