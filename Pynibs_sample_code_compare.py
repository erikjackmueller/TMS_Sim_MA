import os
import h5py
import pynibs
import functions
import numpy as np
from pathlib import Path
path = os.path.realpath(Path("C:/Users/ermu8317/Downloads"))
# path = ""


fn_subject = os.path.join(path, "15484.08_low.hdf5")
fn_roi = os.path.join(path, "15484.08.hdf5")
mesh_idx = "dosing_refined_m1_smooth"
roi_idx = "midlayer_m1s1pmd"
fn_simnibs_results = os.path.join(path, "e.hdf5")
fn_SCSM_results = os.path.join(path, "e_field_midlayer_m1s1_pmd_scsm_10_iter_right_coil.hdf5")
fn_geo = os.path.join(path, 'geo.hdf5')
fn_data_out = os.path.join(path, "ROI_TMS_03_11_22_10_iter_right_coil")

# load subject object
subject = pynibs.load_subject(fn_subject)
mesh_folder = subject.mesh[mesh_idx]["mesh_folder"]

# load roi (midlayer in m1s1pmd)
# roi = pynibs.load_roi_surface_obj_from_hdf5(subject.mesh[mesh_idx]["15484.08"])[roi_idx]
roi = pynibs.load_roi_surface_obj_from_hdf5(fn_roi)[roi_idx]

# calculate/load some BEM results
with h5py.File(fn_SCSM_results, "r") as f:
    e_mag_scsm_roi = f["e"][:]

# load reference e-fields from SimNIBS
with h5py.File(fn_simnibs_results, "r") as f:
    e_mag_simnibs_roi = f["data"]["midlayer"]["roi_surface"]["midlayer_m1s1pmd"]["E_mag"][:]

# calculate difference
e_mag_diff_roi = e_mag_simnibs_roi - e_mag_scsm_roi
print(functions.nrmse(e_mag_simnibs_roi, e_mag_scsm_roi) * 100)

# write data for visualization
pynibs.write_data_hdf5_surf(data=[e_mag_scsm_roi, e_mag_simnibs_roi, e_mag_diff_roi],
                            data_names=["e_mag_scsm_roi", "e_mag_simnibs_roi", "e_mag_diff_roi"],
                            data_hdf_fn_out=fn_data_out,
                            geo_hdf_fn=fn_geo,
                            replace=True,
                            replace_array_in_file=True)