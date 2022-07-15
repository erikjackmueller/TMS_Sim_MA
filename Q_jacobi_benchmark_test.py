from functions import*
import numpy as np
import time
import matplotlib
matplotlib.use("TkAgg")

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
m = d_norm
r_target = sphere_to_carthesian(r=r, phi=phi.flatten(), theta=theta.flatten())

samples = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 500000]
t_numpy = []
t_jacobi = []
errors = []

compare = False

for j in range(2):
    if j == 0:
        ig = False
        print("--------no initial guess---------")
    elif j == 1:
        ig = True
        print("--------with initial guess---------")
    for i in range(len(samples)):


        res1 = reciprocity_three_D(r, theta, r0_v=r0, m=m, phi=phi, projection="sphere_surface")
        tc, areas, tri_points, n_v = sphere_mesh(samples[i], scaling=scaling_factor)
        print(f"elements: {tc.shape[0]}")

        if compare:
            start = time.time()
            Q, rs = SCSM_tri_sphere_numba(tc, tri_points, areas, r0=r0, m=m)
            end = time.time()
            t = t_format(end - start)
            t_numpy.append(t)
            print(f"{t[0]:.2f}" + t[1] + "  Q linalg.solve()")
        else:
            rs = tc

        start = time.time()
        Q = SCSM_jacobi_iter(tc, tri_points, areas, n=n_v, r0=r0, m=m, tol=1e-2, initial_guess=ig)
        end = time.time()
        t = t_format(end - start)
        print(f"{t[0]:.2f}" + t[1] + "  Q jacobi")
        t_jacobi.append(t)
        start = time.time()

        res_flat = SCSM_FMM_E(Q=Q, r_source=rs, r_target=r_target, eps=1e-2, m=m, r0=r0)
        res = array_unflatten(res_flat, n_rows=n)
        res2 = res.copy()
        diff = np.abs(res1 - res2)
        relative_diff = diff / np.linalg.norm(res1)
        rerror_imag = np.linalg.norm(diff) * 100 / np.linalg.norm(res1)
        print(f"relative error: {rerror_imag:.7f}%")
        errors.append(rerror_imag)








