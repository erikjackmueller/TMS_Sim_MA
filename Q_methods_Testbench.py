from functions import*
import numpy as np
import time
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import cm
import tracemalloc
import datetime



small_samples = False

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

n = 100
phi1 = np.linspace(0, np.pi, n)
theta1 = np.linspace(0, 2 * np.pi, n)
phi2, theta2 = np.meshgrid(phi1, theta1)
phi, theta = phi2.T, theta2.T
scaling_factor = 1
r = 0.9
xyz_grid = xyz_grid(r, phi, theta)
# define dipole
direction = np.array([0, 0, 1])
d_norm = direction/np.linalg.norm(direction)
r0 = 1.05 * d_norm
m = np.array([0, 0, 1])
omega = 18.85956e3
r_target = sphere_to_carthesian(r=r, phi=phi.flatten(), theta=theta.flatten())

start = time.time()

# calculate analytic solution
res1 = reciprocity_sphere(grid=xyz_grid, r0_v=r0, m=m, omega=omega)[1]
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " receprocity")
start = time.time()


if small_samples:
    sample_list_small = [2000, 2500, 3000, 3500, 4000]
    length = len(sample_list_small)
    errors = np.zeros((5, length))
    memories = np.zeros_like(errors)
    times = np.zeros_like(errors)
    elements = np.zeros(length)
    for i_samples, samples in enumerate(sample_list_small):
        #tc, areas, tri_points, n_v, avg_len = sphere_mesh(samples)
        tc, areas, tri_points, n_v, avg_len = read_sphere_mesh_from_txt(sizes=None, path="spheres/", n=samples)

        # initialize all methods for JIT reset
        if i_samples == 0:
            Q0 = SCSM_tri_sphere(tc, tri_points, areas, r0=r0, m=m, omega=omega)[0]
            Q1 = SCSM_tri_sphere_numba(tc, tri_points, areas, r0=r0, m=m, omega=omega)[0]
            b_im = jacobi_vectors_numpy(tc, n_v, r0, m, omega=omega)
            Q2 = SCSM_jacobi_iter_numpy(tc, areas, n_v, b_im, tol=5e-16, n_iter=20)
            Q3 = q_jac_vec(tc, areas, n_v, b_im, tol=5e-16, n_iter=20)
            Q4 = q_jac_cu(tc, areas, n_v, b_im, tol=5e-16, n_iter=20)
        print(f"average length: {avg_len:.5f}")
        end = time.time()
        t = t_format(end - start)
        print(f"{t[0]:.2f}" + t[1] + " triangulation")
        n_elements = tc.shape[0]
        print(f"elements: {n_elements}")
        elements[i_samples] = n_elements
        for method in [0, 1, 2, 3, 4]:
            time_0 = start
            tracemalloc.start()
            start = time.time()
            print("--------------------------------------------------")

            if method == 0:
                Q = SCSM_tri_sphere(tc, tri_points, areas, r0=r0, m=m, omega=omega)[0]
                print(f"Loop, number of samples: {samples}")
            elif method == 1:
                Q = SCSM_tri_sphere_numba(tc, tri_points, areas, r0=r0, m=m, omega=omega)[0]
                print(f"numba, number of samples: {samples}")
            elif method == 2:
                b_im = jacobi_vectors_numpy(tc, n_v, r0, m, omega=omega)
                Q = SCSM_jacobi_iter_numpy(tc, areas, n_v, b_im, tol=5e-16, n_iter=20)
                print(f"jacobi, number of samples: {samples}")
            elif method == 3:
                b_im = jacobi_vectors_numpy(tc, n_v, r0, m, omega=omega)
                Q = q_jac_vec(tc, areas, n_v, b_im, tol=5e-16, n_iter=20)
                print(f"vec-jacobi, number of samples: {samples}")
            elif method == 4:
                b_im = jacobi_vectors_numpy(tc, n_v, r0, m, omega=omega)
                Q = q_jac_cu(tc, areas, n_v, b_im, tol=5e-16, n_iter=20)
                print(f"cupy-jacobi, number of samples: {samples}")
            # Q = SCSM_matrix(tc, areas, n=n_v, b_im=b_im, omega=omega)
            mem = tracemalloc.get_traced_memory()[1] / (1024**2)
            tracemalloc.stop()
            print(f"Memory peak: {mem:.4f}MB")
            rs = tc
            end = time.time()
            t = t_format(end - start)
            print(f"{t[0]:.2f}" + t[1] + " Q calc")

            start = time.time()

            b_im_ = vector_potential_for_E_single_m(rs=r_target, m=m, m_pos=r0, omega=omega)
            res_flat = SCSM_FMM_E2(Q=Q, r_source=tc, r_target=r_target, eps=1e-15, b_im=b_im_)
            res = array_unflatten(res_flat, n_rows=n)

            end = time.time()
            t = t_format(end - start)
            print(f"{t[0]:.2f}" + t[1] + "  E calculation")

            time_last = end
            t_full = time_last - time_0
            t = t_format(end - time_0)
            print(f"{t[0]:.2f}" + t[1] + "  complete simulation")


            #
            nrmse_val = nrmse(res1, res) * 100
            print(f"nrmse: {nrmse_val:.7f}%")
            # plot_E_sphere_surf_diff(res1, res, xyz_grid=xyz_grid, c_map=cm.jet)
            errors[method, i_samples] = nrmse_val
            times[method, i_samples] = t_full
            memories[method, i_samples] = mem

    np.savetxt("Q_benchmark_error", errors, delimiter=",")
    np.savetxt("Q_benchmark_memory", memories, delimiter=",")
    np.savetxt("Q_benchmark_time", times, delimiter=",")
    np.savetxt("Q_benchmark_elements", elements, delimiter=",")
    print(f"saved results of elements {elements, memories, times}")
else:
    sample_list_large = [10000, 20000, 40000, 100000, 200000]
    length = len(sample_list_large)
    errors = np.zeros((2, length))
    memories = np.zeros_like(errors)
    times = np.zeros_like(errors)
    elements = np.zeros(length)
    for i_samples, samples in enumerate(sample_list_large): #
        # tc, areas, tri_points, n_v, avg_len = sphere_mesh(samples)
        tc, areas, tri_points, n_v, avg_len = read_sphere_mesh_from_txt(sizes=None, path="spheres/", n=samples)
        # initialize all methods for JIT reset
        if i_samples == 0:
            b_im = jacobi_vectors_numpy(tc, n_v, r0, m, omega=omega)
            Q3 = q_jac_vec(tc, areas, n_v, b_im, tol=5e-16, n_iter=20)
            Q4 = q_jac_cu(tc, areas, n_v, b_im, tol=5e-16, n_iter=20)
        print(f"average length: {avg_len:.5f}")
        end = time.time()
        t = t_format(end - start)
        print(f"{t[0]:.2f}" + t[1] + " triangulation")
        n_elements = tc.shape[0]
        print(f"elements: {n_elements}")
        elements[i_samples] = n_elements
        for method in [0, 1]:
            time_0 = start
            tracemalloc.start()
            start = time.time()
            print("--------------------------------------------------")

            if method == 0:
                b_im = jacobi_vectors_numpy(tc, n_v, r0, m, omega=omega)
                Q = q_jac_vec(tc, areas, n_v, b_im, tol=5e-16, n_iter=20)
                print(f"vec-jacobi, number of samples: {samples}")
            elif method == 1:
                b_im = jacobi_vectors_numpy(tc, n_v, r0, m, omega=omega)
                Q = q_jac_cu(tc, areas, n_v, b_im, tol=5e-16, n_iter=20)
                print(f"cupy-jacobi, number of samples: {samples}")
            # Q = SCSM_matrix(tc, areas, n=n_v, b_im=b_im, omega=omega)
            mem = tracemalloc.get_traced_memory()[1] / (1024 ** 2)
            tracemalloc.stop()
            print(f"Memory peak: {mem:.4f}MB")
            rs = tc
            end = time.time()
            t = t_format(end - start)
            print(f"{t[0]:.2f}" + t[1] + " Q calc")

            start = time.time()

            b_im_ = vector_potential_for_E_single_m(rs=r_target, m=m, m_pos=r0, omega=omega)
            res_flat = SCSM_FMM_E2(Q=Q, r_source=tc, r_target=r_target, eps=1e-15, b_im=b_im_)
            res = array_unflatten(res_flat, n_rows=n)

            end = time.time()
            t = t_format(end - start)
            print(f"{t[0]:.2f}" + t[1] + "  E calculation")

            time_last = end
            t_full = time_last - time_0
            t = t_format(end - time_0)
            print(f"{t[0]:.2f}" + t[1] + "  complete simulation")

            #
            nrmse_val = nrmse(res1, res) * 100
            print(f"nrmse: {nrmse_val:.7f}%")
            # plot_E_sphere_surf_diff(res1, res, xyz_grid=xyz_grid, c_map=cm.jet)
            errors[method, i_samples] = nrmse_val
            times[method, i_samples] = t_full
            memories[method, i_samples] = mem

    np.savetxt("Q_benchmark_large_samples_error", errors, delimiter=",")
    np.savetxt("Q_benchmark_large_samples_memory", memories, delimiter=",")
    np.savetxt("Q_benchmark_large_samples_time", times, delimiter=",")
    np.savetxt("Q_benchmark_large_samples_elements", elements, delimiter=",")
    print(f"saved results of elements {elements}")




