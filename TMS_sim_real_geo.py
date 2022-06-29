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
fn = os.path.join(path, "15484.08.hdf5")

start = time.time()
time_0 = start
tc, areas, tri_points = read_mesh_from_hdf5(fn)
end = time.time()
print(f"{end - start:.2f}s triangulation")

start = time.time()
# Q, rs = functions.SCSM_tri_sphere(tc, areas, r0=r0, m=m, sig=1)
Q, rs = functions.SCSM_tri_sphere_numba(tc, tri_points, areas, r0=r0, m=m)
end = time.time()
print(f"{end - start:.2f}s  Q calculation")