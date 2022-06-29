import functions
import numpy as np
import time
import os
import matplotlib
matplotlib.use("TkAgg")
import multiprocessing.managers
from pathlib import Path
from functions import*

# path_string_original = "C:\Users\ermu8317\Downloads" #cannot be handled by python
# path_string = path_string_original.replace("\\", "/")
path = os.path.realpath(Path("C:/Users/ermu8317/Downloads"))
fn = os.path.join(path, "15484.08.hdf5")

tc, areas = read_mesh_from_hdf5(fn)