import numpy as np
from scipy.special import legendre
from itertools import product
from functools import partial
from mpl_toolkits import mplot3d
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import math
import os
import time
import multiprocessing
import multiprocessing.managers

def read_sphere_mesh_from_txt(sizes, path):

    files = ["con_1_" + str(sizes[0]), "con_2_" + str(sizes[0]), "con_3_" + str(sizes[0]), "x_" + str(sizes[0]),
             "y_" + str(sizes[0]), "z_" + str(sizes[0])]
    connections = np.zeros([3, sizes[1]])
    locations = np.zeros([3, sizes[0]])
    for i in range(6):
        if i < 3:
            connections[i, :] = np.genfromtxt(os.path.join(path, files[i] + ".txt"), dtype=int)
        else:
            locations[i-3, :] = np.genfromtxt(os.path.join(path, files[i] + ".txt"), dtype=float)
    trangle_centers = np.zeros([len(connections[0, :]), 3])
    areas = np.zeros(len(connections[0, :]))



    # calculate centerpoints of trinagles from connections and vertexes
    # center is (AB + BC + CA) / 3 starting from A
    # for a flat trangle in space that should be (x1 + x2 + x3)/3, (y1 + y2 + y3)/3, (z1 + z2 + z3)/3
    # for the area the formula is S = 1/2|AB x AC|, x is the crossproduct in this case
    point1 = np.zeros(3)
    point2 = np.zeros(3)
    point3 = np.zeros(3)

    # plot_mesh(locations, connections, 4, 14)
    # ax1 = plt.axes(projection='3d')
    # plot_triangle(ax1, locations, connections, 4)
    # plot_triangle(ax1, locations, connections, 2) # they're only almost the same
    # plot_triangle(ax1, locations, connections, 12)


    for i in range(len(connections[0, :])):
        p1 = locations[:, int(connections[0, i]) - 1]
        p2 = locations[:, int(connections[1, i]) - 1]
        p3 = locations[:, int(connections[2, i]) - 1]
        p_c_1 = (p1[0] + p2[0] + p3[0]) / 3
        p_c_2 = (p1[1] + p2[1] + p3[1]) / 3
        p_c_3 = (p1[2] + p2[2] + p3[2]) / 3
        trangle_centers[i, :] = np.array([p_c_1, p_c_2, p_c_3])
        line1_2 = p2 - p1
        line1_3 = p3 - p1
        areas[i] = 0.5*vnorm(np.cross(line1_2, line1_3))

    return trangle_centers, areas

def read_sphere_mesh_from_txt_locations_only(sizes, path):
    """
    The locations from a .txt file are triangulated using Delaunay algorithm
    for that to work the 3D carthesian locations are converted to spherical coordinates
    after the the triangulation happens on the "theta-phi_plane"/ the surface of the sphere

    :param sizes: list of sizes [file1_size, file2_size]
    :param path: path to the .txt files
    :return: triangle_centers, areas
    """

    files = ["x_" + str(sizes[0]), "y_" + str(sizes[0]), "z_" + str(sizes[0])]
    locations = np.zeros([3, sizes[0]])

    for i in range(3):
        locations[i, :] = np.genfromtxt(os.path.join(path, files[i] + ".txt"), dtype=float)
    sphere_surf_locations = carthesian_to_sphere(locations.T)[:, 1:]
    # idx [:, 1:] leaves out r which is constant if all is right
    tri = Delaunay(sphere_surf_locations)
    connections = tri.simplices.copy().T

    triangle_centers = np.zeros([len(connections[0, :]), 3])
    areas = np.zeros(len(connections[0, :]))

    # calculate centerpoints of trinagles from connections and vertexes
    # center is (AB + BC + CA) / 3 starting from A
    # for a flat trangle in space that should be (x1 + x2 + x3)/3, (y1 + y2 + y3)/3, (z1 + z2 + z3)/3
    # for the area the formula is S = 1/2|AB x AC|, x is the crossproduct in this case


    for i in range(len(connections[0, :])):
        p1 = locations[:, int(connections[0, i])]
        p2 = locations[:, int(connections[1, i])]
        p3 = locations[:, int(connections[2, i])]
        p_c_1 = (p1[0] + p2[0] + p3[0]) / 3
        p_c_2 = (p1[1] + p2[1] + p3[1]) / 3
        p_c_3 = (p1[2] + p2[2] + p3[2]) / 3
        triangle_centers[i, :] = np.array([p_c_1, p_c_2, p_c_3])
        line1_2 = p2 - p1
        line1_3 = p3 - p1
        areas[i] = 0.5*vnorm(np.cross(line1_2, line1_3))

    # plot_mesh(locations, connections, 0, 10, centers=triangle_centers)
    # ax1 = plt.axes(projection='3d')
    # plot_triangle(ax1, locations, connections, 4, centers=triangle_centers)

    return triangle_centers, areas



def plot_mesh(locations, connections, n1, n2, centers=None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(n1, n2):
        c1 = int(connections[0, i])
        c2 = int(connections[1, i])
        c3 = int(connections[2, i])
        xdata = np.array([locations[0, c1], locations[0, c2], locations[0, c3], locations[0, c1]])
        ydata = np.array([locations[1, c1], locations[1, c2], locations[1, c3], locations[1, c1]])
        zdata = np.array([locations[2, c1], locations[2, c2], locations[2, c3], locations[2, c1]])


        ax.scatter3D(xdata, ydata, zdata)
        ax.plot3D(xdata, ydata, zdata)
    if centers.dtype == 'float64':
        xc = centers[n1:n2, 0]
        yc = centers[n1:n2, 1]
        zc = centers[n1:n2, 2]

        ax.scatter3D(xc, yc, zc, marker='*')
    plt.show()

def plot_triangle(ax, locations, connections, idxs, centers=None):

    i = idxs - 1
    c1 = int(connections[0, i])
    c2 = int(connections[1, i])
    c3 = int(connections[2, i])
    xdata = np.array([locations[0, c1], locations[0, c2], locations[0, c3], locations[0, c1]])
    ydata = np.array([locations[1, c1], locations[1, c2], locations[1, c3], locations[1, c1]])
    zdata = np.array([locations[2, c1], locations[2, c2], locations[2, c3], locations[2, c1]])
    if centers.dtype == 'float64':
        xc = centers[i, 0]
        yc = centers[i, 1]
        zc = centers[i, 2]
        ax.scatter3D(xc, yc, zc, marker='*')
    ax.scatter3D(xdata, ydata, zdata)
    ax.plot3D(xdata, ydata, zdata)
    plt.show()

def func_two_D_dist(r, theta, theta0=0,r0=0.5, sigma=1):
    # r_0 = r0*np.ones(r.shape[0])
    r_out = np.zeros([r.shape[0], theta.shape[0]])
    x0, y0 = np.ones(r.shape[0])*r0*np.cos(theta0), np.ones(r.shape[0])*r0*np.sin(theta0)
    for i_theta in range(theta.shape[0]):
        xs, ys = r*np.cos(theta[i_theta]), r*np.sin(theta[i_theta])
        r_out[:, i_theta] = np.sqrt((xs - x0)**2 + (ys - y0)**2)


    return (r_out + 1e-15)

def vnorm(x):
    return np.linalg.norm(x)

def vangle(x1, x2):
    u_x1 = x1 / np.linalg.norm(x1)
    u_x2 = x2 / np.linalg.norm(x2)
    return np.arccos(np.dot(u_x1, u_x2))

def reciprocity_three_D(r_sphere, theta, r0_v=np.array([12, 0, 0]), m=np.array([0, -1, 0]),phi=0*np.pi, omega=1):
    mu0 = 4*np.pi*1e-7
    E = np.zeros([r_sphere.shape[0], theta.shape[0]])
    r0 = vnorm(r0_v)
    for i_theta in range(theta.shape[0]):
        # xs, ys, zs = r_sphere*np.sin(theta[i_theta]) * np.cos(phi),\
        #              r_sphere*np.sin(theta[i_theta]) * np.sin(phi), r_sphere*np.cos(phi)
        # rs = np.array([xs, ys, zs]) # r is now in carthesian coordinates!
        xs, ys = r_sphere * np.cos(theta[i_theta]), r_sphere * np.sin(theta[i_theta])
        rs = np.array([xs, ys, np.zeros(xs.shape[0])])
        for i_r in range(r_sphere.shape[0]):
            r_v = rs[:, i_r]
            a_v = r0_v - r_v
            a = vnorm(a_v)
            F = (r0*a + np.dot(r0_v, a_v))*a
            nab_F = ((a**2/r0**2) + 2*a + 2*r0 + (np.dot(r0_v, a_v)/a))*r0_v - (a + 2*r0 + (np.dot(r0_v, a_v)/a))*r_v
            E_v = omega*mu0/(4*np.pi*F**2) * (F*np.cross(r_v, m) - np.dot(m, nab_F)*np.cross(r_v, r0_v))
            E[i_r, i_theta] = vnorm(E_v)
    return E

def func_3_shells(r_sphere, theta, r0_v=np.array([12, 0, 0]), r_shells = np.array([7, 7.5, 8]),
                  sigmas=np.array([0.33, 0.01, 0.43])):
    # this is a function for the magnetic filed induced by an electric dipole (antenna for example!)
    H = np.zeros([r_sphere.shape[0], theta.shape[0]])
    r0 = vnorm(r0_v)
    a, b, c = r_shells[0], r_shells[1], r_shells[2]
    s1, s2, s3 = sigmas[0], sigmas[1], sigmas[2]
    for i_theta in range(theta.shape[0]):
        xs, ys = r_sphere * np.cos(theta[i_theta]), r_sphere * np.sin(theta[i_theta])
        rs = np.array([xs, ys, np.zeros(xs.shape[0])])  # r is now in carthesian coordinates
        for i_r in range(r_sphere.shape[0]):
            r_v = rs[:, i_r]
            r = vnorm(r_v)
            gamma = vangle(r_v, r0_v)
            H_i = np.zeros(3)
            for l in range(1, 4):
                P = legendre(l)(np.cos(gamma))
                # l is the subscript for the order of the polynomial
                # cos(gamma) is the argument
                A = ((2*l + 1)**3 / 2*l) / (((s1/s2 + 1)*l + 1)*((s2/s3 + 1)*l + 1) +
                                            (s1/s2 - 1)*(s2/s3 - 1)*l*(l+1)*(a/b)**(2*l + 1) +
                                            (s2/s3 - 1)*(l + 1)*((s1/s2 + 1)*l + 1)*(b/c)**(2*l + 1) +
                                            (s1/s2 - 1)*(l + 1)*((s2/s3 + 1)*(l + 1) - 1)*(a/c)**(2*l + 1))
                H_i[l-1] = A * (r0**l/c**(l + 1)) * P

            H[i_r, i_theta] = 1 / (2*np.pi*s3) * np.sum(H_i)
    return H

def v(n):
    return (1/2)*(-1 + np.sqrt(1 + 4*n*(n + 1)))

def P(n, v, r):
    return r**v(n)
def P_prime(n, v, r):
    return v(n)*r**(v(n) - 1)
def Q(n, v, r):
    return r**(-v(n) - 1)
def Q_prime(n, v, r):
    return (-v(n) - 1)*r**(-v(n) - 2)
def Y_real(l, m, alpha, theta, phi=0):
    c = np.sqrt((2*l + 1) / (np.pi*4) * (math.factorial(l-m)/math.factorial(l+m)))
    return c*legendre(l,m)(np.cos(theta))*np.cos(m*phi)

def func_de_Munck_potential(rs, theta, r0_v=np.array([12, 0, 0]), m=np.array([0, 1, 0]), r_shells = np.array([7, 7.5, 8]),
                  sigmas=np.array([0.43, 0.01, 0.33]), n=5):
    #sigmas from outside to inside
    phi = np.zeros([n, rs.shape[0], theta.shape[0]])
    r_0 = vnorm(r0_v)
    m_r = vnorm(m)
    m_theta = np.arccos(m[2]/m_r)
    r0, r1, r2 = r_shells[2], r_shells[1], r_shells[0]
    A = np.zeros([2, 3])
    B = np.zeros([2, 3])
    A[0, 2], B[0, 2] = 1, 0
    for i_n in range(n):
        # A^1_2 = A[0, 1]
        # change 1 -> 0; add [ ] in arrays
        AB_0_1 = np.dot(np.dot(np.linalg.inv(np.array([[P(n, v, r1), Q(n, v, r1)], [sigmas[1] * P_prime(n, v, r1),
                                                                            sigmas[1] * Q_prime(n, v, r1)]])),
                               np.array([[P(n, v, r2), Q(n, v, r2)], [sigmas[2] * P_prime(n, v, r2),
                                                                     sigmas[2] * Q_prime(n, v, r2)]])),
                        np.array([A[0, 2], B[0, 2]]))
        A[0, 1], B[0, 1] = AB_0_1[0], AB_0_1[1]
        AB_0_0 = np.dot(np.dot(np.linalg.inv(np.array([[P(n, v, r0), Q(n, v, r0)], [sigmas[0] * P_prime(n, v, r0),
                                                                            sigmas[0] * Q_prime(n, v, r0)]])),
                               np.array([[P(n, v, r1), Q(n, v, r1)], [sigmas[1] * P_prime(n, v, r1),
                                                                     sigmas[1] * Q_prime(n, v, r1)]])),
                        np.array([A[0, 1], B[0, 1]]))
        A[0, 0], B[0, 0] = AB_0_0[0], AB_0_0[1]
        A[1, 0] = -Q_prime(n, v, r0)/ P_prime(n, v, r0)
        B[1, 0] = 1
        AB_1_1 = np.dot(np.dot(np.linalg.inv(np.array([[P(n, v, r1), Q(n, v, r1)], [sigmas[1] * P_prime(n, v, r1),
                                                                            sigmas[1] * Q_prime(n, v, r1)]])),
                               np.array([[P(n, v, r0), Q(n, v, r0)], [sigmas[0] * P_prime(n, v, r0),
                                                                     sigmas[0] * Q_prime(n, v, r0)]])),
                        np.array([A[0, 1], B[0, 1]]))
        A[1, 1], B[1, 1] = AB_1_1[0], AB_1_1[1]
        AB_1_2 = np.dot(np.dot(np.linalg.inv(np.array([[P(n, v, r2), Q(n, v, r2)], [sigmas[2] * P_prime(n, v, r2),
                                                                            sigmas[2] * Q_prime(n, v, r2)]])),
                               np.array([[P(n, v, r1), Q(n, v, r1)], [sigmas[1] * P_prime(n, v, r1),
                                                                     sigmas[1] * Q_prime(n, v, r1)]])),
                        np.array([A[0, 1], B[0, 1]]))
        A[1, 2], B[1, 2] = AB_1_2[0], AB_1_2[1]
        for i_theta in range(theta.shape[0]):
            th = theta[i_theta]
            for i_r in range(rs.shape[0]):
                r = rs[i_r]
                if r < r2:
                    R_2 = A[1, 2] * Q(n, v, r) + B[1, 2] * P(n, v, r)
                elif r < r1:
                    R_2 = A[1, 1] * Q(n, v, r) + B[1, 1] * P(n, v, r)
                else:
                    R_2 = A[1, 0] * Q(n, v, r) + B[1, 0] * P(n, v, r)
                R_1 = A[0, 0] * Q(n, v, r_0) + B[0, 0] * P(n, v, r_0)

                phi[i_n, i_theta, i_r] = (R_2/sigmas[2]*B[1, 2])*(m_r*R_1*Y_real(n, 0, 0, th) + m_theta*r_0**-1*R_1*Y_real(n, 1, 0, th))

    return -1/(4*np.pi) * np.sum(phi, axis=0)

def plot_default():
    # take r_head = 8
    # r coil/ m = r0
    # r_shells=np.array([7, 7.5, 8])
    # sigmas=np.array([0.33, 0.01, 0.43]
    r = np.linspace(0.01, 8, 400)
    theta = np.linspace(0, 2*np.pi, 400)
    line1 = 7*np.ones(400)
    line2 = 7.5*np.ones(400)
    line3 = 8*np.ones(400)
    r0 = np.array([12, 0, 0])
    ax = plt.subplot(111, projection='polar')
    res = reciprocity_three_D(r, theta, r0_v=r0, m=np.array([0, 1, 0]))
    # res_mag = func_3_shells(r, theta, r0_v=r0)
    # res = func_de_Munck_potential(r, theta)
    f_min, f_max = res.min(), res.max()
    # f_min, f_max = res_mag.min(), res_mag.max()

    im = ax.pcolormesh(theta, r, res, cmap='plasma', vmin=f_min, vmax=f_max)
    # im = ax.pcolormesh(theta, r, res_mag, cmap='plasma', vmin=f_min, vmax=f_max)
    ax.set_yticklabels([])
    ax.set_rmax(8)
    ax.plot(theta, line1, c='k')
    ax.plot(theta, line2, c='k')
    ax.plot(theta, line3, c='k')
    ax.grid(True)
    ax.set_title(f"|E| for r0 = {str(r0)}^T")

    plt.show()

def plot_E(res, r, theta, r_max):

    fig = plt.figure()
    ax = plt.subplot(111, projection='polar')
    f_min, f_max = res.min(), res.max()
    im = ax.pcolormesh(theta, r, res, cmap='plasma', vmin=f_min, vmax=f_max)
    ax.set_yticklabels([])
    ax.set_rmax(r_max)
    ax.grid(True)
    fig.colorbar(im)
    # ax.set_title(f"|H| for r0 = {str(r0)}^T")

    plt.show()
# plot_default()
def sphere_to_carthesian(r, theta, phi):
    return r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)

def carthesian_to_sphere(r_carth):
    r_sphere = np.zeros(r_carth.shape)
    xy = r_carth[:, 0] ** 2 + r_carth[:, 1] ** 2
    r_sphere[:, 0] = np.sqrt(xy + r_carth[:, 2] ** 2)
    r_sphere[:, 1] = np.arctan2(np.sqrt(xy), r_carth[:, 2])  # for elevation angle defined from Z-axis down
    r_sphere[:, 2] = np.arctan2(r_carth[:, 1], r_carth[:, 0])
    return r_sphere

def trapezoid_area_and_centre(rs):
    b1_v = rs[1, 0, :] - rs[0, 0, :]
    b2_v = rs[1, 1, :] - rs[0, 1, :]
    a_v = rs[0, 1, :] - rs[0, 0, :]
    db_v = 1/2*(b1_v-b2_v)
    b1, b2, db, a = vnorm(b1_v), vnorm(b2_v), vnorm(db_v), vnorm(a_v)
    h = np.sqrt(a**2 - db**2)
    area = 1/2*h*(b1+b2)
    r_center = rs[0, 0, :] + b1_v/2 + 1/2*(a_v + db_v)
    n_v = np.cross(b1_v, a_v)
    return area, r_center, n_v

def kroen(a, b):
    if a == b:
        return 1
    else:
        return 0

def SCSM_trapezes(N=100, r=8, r0 = np.array([0, 0, 11]), m = np.array([0, 1, 0]), sig = 0.33, omega=1):

    eps0 = 8.854187812813e-12
    phis  = np.linspace(0, 2*np.pi, N)
    thetas  = np.linspace(0, 2*np.pi, N)
    M = N**2
    rs = np.zeros([M, 3])
    areas = np.zeros(M)
    rs_trap = np.zeros([2,2,3]) # 2x2 matrix with vectors as entries
    norm_vects = np.zeros([M, 3])
    A_real = np.zeros([M, M])
    A_imag = np.zeros([M, M])
    B = np.zeros(M)
    n = 0
    for i in range(N-1):
        for j in range(N-1):
            rs_trap[0, 0, :] = sphere_to_carthesian(r, thetas[i], phis[j])
            rs_trap[0, 1, :] = sphere_to_carthesian(r, thetas[i+1], phis[j])
            rs_trap[1, 0, :] = sphere_to_carthesian(r, thetas[i], phis[j+1])
            rs_trap[1, 1, :] = sphere_to_carthesian(r, thetas[i+1], phis[j+1])
            areas[n], rs[n,:], norm_vects[n, :] = trapezoid_area_and_centre(rs_trap)
            n+=1

    for u in range(M):
        for v in range(M):
            A1 = 1/(4*np.pi * eps0 * vnorm(rs[u, :] - rs[v, :])**3 + kroen(u, v))*(rs[u, :] - rs[v, :])@norm_vects[v]
            A2 = kroen(u, v)/2*eps0*areas[u]*(1/2 + (omega*eps0/sig)* 1j)
            A = np.array([A1, 0]) - np.array([A2.real, A2.imag])
            A_real[u, v] = A[0]
        B[u] = vnorm(1e-7*(np.cross(m, (rs[u] - r0)))/(vnorm(rs[u] - r0)**3))

    Q = np.linalg.solve(A_real, B)
    return Q, rs

def SCSM_tri_sphere(tri_centers, areas, r0 = np.array([0, 0, 1.1]), m = np.array([0, 1, 0]), sig = 0.33, omega=1):
    rs = tri_centers
    M = rs.shape[0]
    A = np.zeros([M, M], dtype=np.complex_)
    B = np.zeros(M)
    eps0 = 8.854187812813e-12
    for i in range(M):
        for j in range(M):
            r_norm_j = rs[j, :]/vnorm(rs[j])
            A1 = 1/(4*np.pi * eps0 * vnorm(rs[i, :] - rs[j, :])**3 + kroen(i, j))*(rs[i, :] - rs[j, :])@r_norm_j
            A2_real = kroen(i, j)/(2*eps0*areas[i])*(1/2)
            A2_imag = kroen(i, j)/(2*eps0*areas[i])*(omega * eps0 / sig) * 1j
            A[i, j] = A1 + A2_real + A2_imag
        B[i] = vnorm(1e-7*(np.cross(m, (rs[i] - r0)))/(vnorm(rs[i] - r0)**3))

    Q = np.linalg.solve(A, B)
    return Q, rs

# @numba.jit(nopython=True, parallel=True)
# def numba_SCSM_E_sphere(Q, r_q, r_sphere, theta, E, E_v, theta_size, r_sphere_size,
#                         r_q_size, m=np.array([0, 1, 0]), r0=np.array([0, 0, 1.1]), omega=1):
#     eps0 = 8.854187812813e-12
#     mu0 = 4*np.pi*1e-7
#     E_v_1 = np.zeros(r_q_size, dtype=np.complex_)
#     E_v_2 = np.zeros(r_q_size, dtype=np.complex_)
#     E_v_3 = np.zeros(r_q_size, dtype=np.complex_)
#
#     for i_theta in numba.prange(theta_size):
#         xs, ys = r_sphere * np.cos(theta[i_theta]), r_sphere * np.sin(theta[i_theta])
#         rs = np.concatenate((xs, ys, np.zeros(xs.shape[0])), axis=0)  # r is now in carthesian coordinates!
#         for i_r in numba.prange(r_sphere_size):
#             rs_x = rs[0]
#             rs_y = rs[1]
#             rs_z = rs[2]
#             for n in numba.prange(r_q_size):
#                 E_v_1[n] = 1#Q[n] * (rs_x[i_r] - r_q[n]) / (4 * np.pi * eps0 *
#                 np.linalg.norm(rs_x[i_r] - r_q[n]) ** 3) - (1j * omega * mu0) /
#                 (4 * np.pi * np.linalg.norm(rs_x[i_r] - r0) ** 3) * (np.cross(m, (rs_x[i_r] - r0)))
#                 # E_v_2[n] = Q[n] * (rs_y[i_r] - r_q[n]) / (4 * np.pi * eps0 *
#                 np.linalg.norm(rs_y[i_r] - r_q[n]) ** 3) - (1j * omega * mu0) /
#                 (4 * np.pi * np.linalg.norm(rs_y[i_r] - r0) ** 3) * (np.cross(m, (rs_y[i_r] - r0)))
#                 # E_v_3[n] = Q[n] * (rs_z[i_r] - r_q[n]) / (4 * np.pi * eps0 *
#                 np.linalg.norm(rs_z[i_r] - r_q[n]) ** 3) - (1j * omega * mu0) /
#                 (4 * np.pi * np.linalg.norm(rs_z[i_r] - r0) ** 3) * (np.cross(m, (rs_z[i_r] - r0)))
#             E[i_r, i_theta] = np.sqrt(E_v_1.imag.sum(axis=1)**2 + E_v_2.imag.sum(axis=1)**2 /
#             + E_v_3.imag.sum(axis=1)**2)
#             # E[i_r, i_theta] = vnorm(E_v.imag.sum(axis=1)) # alternative for inly imaginary part
#     print(i_theta+1)
#     return E

def SCSM_E_sphere(Q, r_q, r_sphere, theta, m=np.array([0, 1, 0]), r0=np.array([0, 0, 1.1]), omega=1):
    eps0 = 8.854187812813e-12
    mu0 = 4*np.pi*1e-7
    E = np.zeros([r_sphere.shape[0], theta.shape[0]])
    E_v = np.zeros([3, r_q.shape[0]], dtype=np.complex_)
    start_time = time.time()
    time_1 = 0
    for i_theta in range(theta.shape[0]):
        xs, ys = r_sphere * np.cos(theta[i_theta]), r_sphere * np.sin(theta[i_theta])
        rs = np.array([xs, ys, np.zeros(xs.shape[0])])  # r is now in carthesian coordinates!
        for i_r in range(r_sphere.shape[0]):
            r_v = rs[:, i_r]
            for n in range(r_q.shape[0]):
                E_v[:, n] = Q[n]*(r_v - r_q[n])/(4*np.pi*eps0*vnorm(r_v - r_q[n])**3)\
                            - (1j*omega*mu0)/(4*np.pi * vnorm(r_v - r0)**3) * (np.cross(m, (r_v - r0)))
            E[i_r, i_theta] = vnorm(E_v.sum(axis=1))
            # E[i_r, i_theta] = vnorm(E_v.imag.sum(axis=1)) # alternative for inly imaginary part
        if i_theta == 0:
            time_1 = time.time()
    print(f"angle {i_theta + 1} of {theta.shape[0]} done, "
          f"{format(((time_1 - start_time)*theta.shape[0] - (time.time() - start_time))/60, '.1f')}")
    return E

def E_parallel(idxs, E, Q, r_sphere, r_q, r0, theta, m, phi, omega, eps0, mu0, N):
    for k in range(len(idxs)):
        i = idxs[k][0]
        j = idxs[k][1]
        xs, ys = r_sphere * np.cos(theta[j]), r_sphere * np.sin(theta[j])
        rs = np.array([xs, ys, np.zeros(xs.shape[0])])
        r_v = rs[:, i]
        grad_phi = np.zeros([3, N], dtype=np.complex_)
        for n in range(N):
            grad_phi[:, n] = Q[n] * (r_v - r_q[n]) / (4 * np.pi * eps0 * vnorm(r_v - r_q[n]) ** 3)
        E_complex = grad_phi.sum(axis=1) - (1j * omega * mu0) / (4 * np.pi * vnorm(r_v - r0) ** 3)\
                    * (np.cross(m, (r_v - r0)))
        E[i, j] = vnorm(E_complex.imag)

def parallel_SCSM_E_sphere(manager, Q, r_q, r_sphere, theta, m=np.array([0, 1, 0]), r0=np.array([0, 0, 1.1]),
                           phi=0, omega=1):
    eps0 = 8.854187812813e-12
    mu0 = 4*np.pi*1e-7
    # E_v = np.zeros([3, r_q.shape[0]], dtype=np.complex_)

    I, J, N = r_sphere.shape[0], theta.shape[0], r_q.shape[0]
    manager.start()
    E = manager.np_zeros([I, J])
    # n_cpu = 5
    n_cpu = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(n_cpu)
    idx_sequence = product(list(range(I)), list(range(J)))
    idx_list = list(idx_sequence)

    workhorse_partial = partial(E_parallel, E=E, Q=Q, r_sphere=r_sphere, r_q=r_q, r0=r0, theta=theta,
                                m=m, phi=phi, omega=omega, eps0=eps0, mu0=mu0, N=N)
    # for i in range(I):
    #     for j in range(J):
    #
    #         xs, ys = r_sphere * np.cos(theta[j]), r_sphere * np.sin(theta[j])
    #         rs = np.array([xs, ys, np.zeros(xs.shape[0])])
    #         r_v = rs[:, i]
    #         grad_phi = np.zeros([3, N], dtype=np.complex_)
    #         for n in range(N):
    #             grad_phi[:, n] = Q[n] * (r_v - r_q[n]) / (4 * np.pi * eps0 * vnorm(r_v - r_q[n]) ** 3)
    #         E_complex = grad_phi.sum(axis=1) - (1j * omega * mu0) / (4 * np.pi * vnorm(r_v - r0) ** 3) * (np.cross(m, (r_v - r0)))
    #         E[i, j] = vnorm(E_complex.imag)

    idx_list_chunks = compute_chunks(idx_list, n_cpu)
    pool.map(workhorse_partial, idx_list_chunks)

    return np.array(E)


# r = 8
# theta, phi  = np.meshgrid(np.linspace(0, 2*np.pi, 100), np.linspace(0, 2*np.pi, 100))
# fig = plt.figure()
# xs, ys, zs = r*np.cos(phi)*np.sin(theta),  r*np.sin(phi)*np.sin(theta),  r*np.cos(theta)
# ax = plt.subplot(111, projection='3d')
# ax.scatter(xs, ys, zs)
# plt.show()




def compute_chunks(seq, num):
    """
    Splits up a sequence _seq_ into _num_ chunks of similar size.
    If len(seq) < num, (num-len(seq)) empty chunks are returned so that len(out) == num
    Parameters
    ----------
    seq : list of something [N_ele]
        List containing data or indices, which is divided into chunks
    num : int
        Number of chunks to generate
    Returns
    -------
    out : list of num sublists
        num sub-lists of seq with each of a similar number of elements (or empty).
    """
    assert len(seq) > 0
    assert num > 0
    # assert isinstance(seq, list), f"{type(seq)} can't be chunked. Provide list."

    avg = len(seq) / float(num)
    n_empty = 0  # if len(seg) < num, how many empty lists to append to return?

    if avg < 1:
        avg = 1
        n_empty = num - len(seq)

    out = []
    last = 0.0

    while last < len(seq):
        # if only one element would be left in the last run, add it to the current
        if (int(last + avg) + 1) == len(seq):
            last_append_idx = int(last + avg) + 1
        else:
            last_append_idx = int(last + avg)

        out.append(seq[int(last):last_append_idx])

        if (int(last + avg) + 1) == len(seq):
            last += avg + 1
        else:
            last += avg

    # append empty lists if len(seq) < num
    out += [[]] * n_empty

    return out

