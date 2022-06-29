from typing import Union, Any

import numpy as np
from scipy.special import legendre
from itertools import product
from functools import partial
# from mpl_toolkits import mplot3d
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib import cm
import h5py
import numba
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
            locations[i - 3, :] = np.genfromtxt(os.path.join(path, files[i] + ".txt"), dtype=float)
    trangle_centers = np.zeros([len(connections[0, :]), 3])
    areas = np.zeros(len(connections[0, :]))

    # calculate centerpoints of trinagles from connections and vertexes
    # center is (AB + BC + CA) / 3 starting from A
    # for a flat trangle in space that should be (x1 + x2 + x3)/3, (y1 + y2 + y3)/3, (z1 + z2 + z3)/3
    # for the area the formula is S = 1/2|AB x AC|, x is the crossproduct in this case

    # plot_mesh(locations, connections, 0, 500)
    # ax1 = plt.axes(projection='3d')
    # plot_triangle(ax1, locations, connections, 4, centers=trangle_centers)
    # plot_triangle(ax1, locations, connections, 2) # they're only almost the same
    # plot_triangle(ax1, locations, connections, 12)

    for i in range(len(connections[0, :])):
        p1 = locations[:, int(connections[0, i])]
        p2 = locations[:, int(connections[1, i])]
        p3 = locations[:, int(connections[2, i])]
        p_c_1 = (p1[0] + p2[0] + p3[0]) / 3
        p_c_2 = (p1[1] + p2[1] + p3[1]) / 3
        p_c_3 = (p1[2] + p2[2] + p3[2]) / 3
        trangle_centers[i, :] = np.array([p_c_1, p_c_2, p_c_3])
        line1_2 = p2 - p1
        line1_3 = p3 - p1
        areas[i] = 0.5 * vnorm(np.cross(line1_2, line1_3))

    return trangle_centers, areas


def read_sphere_mesh_from_txt_locations_only(sizes, path, tri_points=False, scaling=1):
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
    locations = scaling * locations
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

    n = len(connections[0, :])
    triangle_points = np.zeros((n, 3, 3))
    for i in range(n):
        p1 = locations[:, int(connections[0, i])]
        p2 = locations[:, int(connections[1, i])]
        p3 = locations[:, int(connections[2, i])]
        triangle_points[i][:][:] = np.vstack((p1, p2, p3))
        p_c_1 = (p1[0] + p2[0] + p3[0]) / 3
        p_c_2 = (p1[1] + p2[1] + p3[1]) / 3
        p_c_3 = (p1[2] + p2[2] + p3[2]) / 3
        triangle_centers[i, :] = np.array([p_c_1, p_c_2, p_c_3])
        line1_2 = p2 - p1
        line1_3 = p3 - p1
        areas[i] = 0.5 * vnorm(np.cross(line1_2, line1_3))
    # plot_mesh(locations, connections, 0, 1200, centers=triangle_centers)
    # ax1 = plt.axes(projection='3d')
    # plot_triangle(ax1, locations, connections, 4, centers=triangle_centers)
    if tri_points:
        return triangle_centers, areas, triangle_points
    else:
        return triangle_centers, areas

def read_mesh_from_hdf5(fn):
    with h5py.File(fn, "r") as f:
        a_group_key = list(f.keys())[0]
        key_list = list(f.keys())

        mesh_data = f['mesh']
        elm_number = np.array(f['mesh']['elm']['elm_number'])
        elm_type = np.array(f['mesh']['elm']['elm_type'])
        node_number_list = np.array(f['mesh']['elm']['node_number_list'])
        tri_tissue_type = np.array(f['mesh']['elm']['tri_tissue_type'])
        triangle_number_list = np.array(f['mesh']['elm']['triangle_number_list'])
        node_number = np.array(f['mesh']['nodes']['node_number'])
        node_coords = np.array(f['mesh']['nodes']['node_coord'])

    elms_wm = elm_number[np.where(elm_type == 2)]
    tris_wm = triangle_number_list[np.where(elm_type == 2)]
    n = tris_wm.shape[0]
    triangle_centers = np.zeros([n, 3])
    areas = np.zeros(n)

    # calculate centerpoints of trinagles from connections and vertexes
    # center is (AB + BC + CA) / 3 starting from A
    # for a flat trangle in space that should be (x1 + x2 + x3)/3, (y1 + y2 + y3)/3, (z1 + z2 + z3)/3
    # for the area the formula is S = 1/2|AB x AC|, x is the crossproduct in this case

    triangle_points = np.zeros((n, 3, 3))
    for i in range(n):
        p1 = node_coords[int(tris_wm[i, 0]), :]
        p2 = node_coords[int(tris_wm[i, 1]), :]
        p3 = node_coords[int(tris_wm[i, 2]), :]
        triangle_points[i][:][:] = np.vstack((p1, p2, p3))
        p_c_1 = (p1[0] + p2[0] + p3[0]) / 3
        p_c_2 = (p1[1] + p2[1] + p3[1]) / 3
        p_c_3 = (p1[2] + p2[2] + p3[2]) / 3
        triangle_centers[i, :] = np.array([p_c_1, p_c_2, p_c_3])
        line1_2 = p2 - p1
        line1_3 = p3 - p1
        areas[i] = 0.5 * np.linalg.norm(np.cross(line1_2, line1_3))
    plot_mesh(node_coords.T, tris_wm.T, 0, n, centers=triangle_centers)
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


def func_two_D_dist(r, theta, theta0=0, r0=0.5, sigma=1):
    # r_0 = r0*np.ones(r.shape[0])
    r_out = np.zeros([r.shape[0], theta.shape[0]])
    x0, y0 = np.ones(r.shape[0]) * r0 * np.cos(theta0), np.ones(r.shape[0]) * r0 * np.sin(theta0)
    for i_theta in range(theta.shape[0]):
        xs, ys = r * np.cos(theta[i_theta]), r * np.sin(theta[i_theta])
        r_out[:, i_theta] = np.sqrt((xs - x0) ** 2 + (ys - y0) ** 2)

    return (r_out + 1e-15)


def vnorm(x):
    return np.linalg.norm(x)


def vangle(x1, x2):
    u_x1 = x1 / np.linalg.norm(x1)
    u_x2 = x2 / np.linalg.norm(x2)
    return np.arccos(np.dot(u_x1, u_x2))


def reciprocity_three_D(r_sphere, theta, r0_v=np.array([12, 0, 0]), m=np.array([0, -1, 0]), phi=0 * np.pi, omega=1,
                        projection="polar"):
    mu0 = 4 * np.pi * 1e-7
    r0 = vnorm(r0_v)
    if projection == "polar":
        E = np.zeros([r_sphere.shape[0], theta.shape[0]])
        for i_theta in range(theta.shape[0]):
            for i_r in range(r_sphere.shape[0]):

                    xs, ys = r_sphere * np.cos(theta[i_theta]), r_sphere * np.sin(theta[i_theta]) # r is now in carthesian coordinates!
                    rs = np.array([xs, ys, np.zeros(xs.shape[0])])
                    r_v = rs[:, i_r]

                    a_v = r0_v - r_v
                    a = vnorm(a_v)
                    F = (r0 * a + np.dot(r0_v, a_v)) * a
                    nab_F = ((a ** 2 / r0 ** 2) + 2 * a + 2 * r0 + (np.dot(r0_v, a_v) / a)) * r0_v - (
                                a + 2 * r0 + (np.dot(r0_v, a_v) / a)) * r_v
                    E_v = omega * mu0 / (4 * np.pi * F ** 2) * (F * np.cross(r_v, m) - np.dot(m, nab_F) * np.cross(r_v, r0_v))
                    E[i_r, i_theta] = vnorm(E_v)
    elif projection == "sphere_surface":
        E = np.zeros([phi.shape[0], theta.shape[0]])
        for i_phi in range(phi.shape[0]):
            for i_theta in range(theta.shape[0]):
                phi_i = phi[i_phi, 0]
                theta_j = theta[0, i_theta]
                x, y, z = r_sphere * np.sin(phi_i) * np.cos(theta_j), r_sphere * np.sin(phi_i) * np.sin(theta_j),\
                          r_sphere * np.cos(phi_i)
                r_v = np.array([x, y, z])
                a_v = r0_v - r_v
                a = vnorm(a_v)
                F = (r0 * a + np.dot(r0_v, a_v)) * a
                nab_F = ((a ** 2 / r0 ** 2) + 2 * a + 2 * r0 + (np.dot(r0_v, a_v) / a)) * r0_v - (
                        a + 2 * r0 + (np.dot(r0_v, a_v) / a)) * r_v
                E_v = omega * mu0 / (4 * np.pi * F ** 2) * (
                            F * np.cross(r_v, m) - np.dot(m, nab_F) * np.cross(r_v, r0_v))
                E[i_phi, i_theta] = vnorm(E_v)
    else:
        raise TypeError("projection can only be 'polar' or 'sphere_surface'!")
    return E


def func_3_shells(r_sphere, theta, r0_v=np.array([12, 0, 0]), r_shells=np.array([7, 7.5, 8]),
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
                A = ((2 * l + 1) ** 3 / 2 * l) / (((s1 / s2 + 1) * l + 1) * ((s2 / s3 + 1) * l + 1) +
                                                  (s1 / s2 - 1) * (s2 / s3 - 1) * l * (l + 1) * (a / b) ** (2 * l + 1) +
                                                  (s2 / s3 - 1) * (l + 1) * ((s1 / s2 + 1) * l + 1) * (b / c) ** (
                                                              2 * l + 1) +
                                                  (s1 / s2 - 1) * (l + 1) * ((s2 / s3 + 1) * (l + 1) - 1) * (a / c) ** (
                                                              2 * l + 1))
                H_i[l - 1] = A * (r0 ** l / c ** (l + 1)) * P

            H[i_r, i_theta] = 1 / (2 * np.pi * s3) * np.sum(H_i)
    return H


def v(n):
    return (1 / 2) * (-1 + np.sqrt(1 + 4 * n * (n + 1)))


def P(n, v, r):
    return r ** v(n)


def P_prime(n, v, r):
    return v(n) * r ** (v(n) - 1)


def Q(n, v, r):
    return r ** (-v(n) - 1)


def Q_prime(n, v, r):
    return (-v(n) - 1) * r ** (-v(n) - 2)


def Y_real(l, m, alpha, theta, phi=0):
    c = np.sqrt((2 * l + 1) / (np.pi * 4) * (math.factorial(l - m) / math.factorial(l + m)))
    return c * legendre(l, m)(np.cos(theta)) * np.cos(m * phi)


def func_de_Munck_potential(rs, theta, r0_v=np.array([12, 0, 0]), m=np.array([0, 1, 0]), r_shells=np.array([7, 7.5, 8]),
                            sigmas=np.array([0.43, 0.01, 0.33]), n=5):
    # sigmas from outside to inside
    phi = np.zeros([n, rs.shape[0], theta.shape[0]])
    r_0 = vnorm(r0_v)
    m_r = vnorm(m)
    m_theta = np.arccos(m[2] / m_r)
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
        A[1, 0] = -Q_prime(n, v, r0) / P_prime(n, v, r0)
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

                phi[i_n, i_theta, i_r] = (R_2 / sigmas[2] * B[1, 2]) * (
                            m_r * R_1 * Y_real(n, 0, 0, th) + m_theta * r_0 ** -1 * R_1 * Y_real(n, 1, 0, th))

    return -1 / (4 * np.pi) * np.sum(phi, axis=0)


def plot_default():
    # take r_head = 8
    # r coil/ m = r0
    # r_shells=np.array([7, 7.5, 8])
    # sigmas=np.array([0.33, 0.01, 0.43]
    r = np.linspace(0.01, 8, 400)
    theta = np.linspace(0, 2 * np.pi, 400)
    line1 = 7 * np.ones(400)
    line2 = 7.5 * np.ones(400)
    line3 = 8 * np.ones(400)
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


def plot_E_diff(res1, res2, r, theta, r_max, r0=None, m=None):
    diff = np.abs(res2 - res1)
    fig, ax = plt.subplots(1, 4, subplot_kw={'projection': 'polar'}, figsize=(18, 4))
    ax0, ax1, ax2, ax3 = ax[0], ax[1], ax[2], ax[3]
    im1 = ax0.pcolormesh(theta, r, res1, cmap='plasma', vmin=res1.min(), vmax=res1.max())
    fig.colorbar(im1, ax=ax0)
    ax0.set_yticklabels([])
    ax0.set_ylim(0, r_max)
    # ax0.set_yticks(np.arange(0, r_max, r_max / 10))
    ax0.grid(True)
    f_max = max(res1.max(), res2.max())
    f_min = min(res1.min(), res2.min())
    im = ax1.pcolormesh(theta, r, res1, cmap='plasma', vmin=f_min, vmax=f_max)
    ax1.set_yticklabels([])
    ax1.set_ylim(0, r_max)
    # ax1.set_yticks(np.arange(0, r_max, r_max/10))
    ax1.grid(True)
    im = ax2.pcolormesh(theta, r, res2, cmap='plasma', vmin=f_min, vmax=f_max)
    ax2.set_yticklabels([])
    ax2.set_ylim(0, r_max)
    # ax2.set_yticks(np.arange(0, r_max, r_max / 10))
    ax2.grid(True)
    im = ax3.pcolormesh(theta, r, diff, cmap='plasma', vmin=f_min, vmax=f_max)
    ax3.set_yticklabels([])
    ax3.set_ylim(0, r_max)
    # ax3.set_yticks(np.arange(0, r_max, r_max / 10))
    ax3.grid(True)
    fig.colorbar(im)
    ax0.set_title("analytic (original scale)")
    ax1.set_title("analytic (scaled to max-value)")
    ax2.set_title("numeric")
    ax3.set_title("difference")
    plt.subplots_adjust(wspace=0.7)
    rerror = np.linalg.norm(diff) / np.linalg.norm(res1)
    fig.suptitle(f"relative error: {rerror:.6f}, r0 = {r0}, m = {m}")

    plt.show()


def plot_E_sphere_surf(res, phi, theta, r):
    p, t = np.meshgrid(phi, theta)
    x = np.cos(p) * np.sin(t)
    y = np.sin(p) * np.sin(t)
    z = np.cos(t)
    fcolors = res
    fmax, fmin = fcolors.max(), fcolors.min()
    fcolors = (fcolors - fmin) / (fmax - fmin)
    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.plasma(fcolors))
    # ax.grid()
    m = cm.ScalarMappable(cmap=cm.plasma)
    fig.colorbar(m, shrink=0.5, pad=0.15)
    plt.show()

def plot_E_sphere_surf_diff(res1, res2, phi, theta, r):
    diff = np.abs(res2 - res1)
    fig, ax = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(12, 4))
    ax1, ax2, ax3 = ax[0], ax[1], ax[2]
    p, t = np.meshgrid(phi, theta)
    x = np.cos(p) * np.sin(t)
    y = np.sin(p) * np.sin(t)
    z = np.cos(t)
    fcolors1 = res1
    fmax, fmin = fcolors1.max(), fcolors1.min()
    fcolors1 = (fcolors1 - fmin) / (fmax - fmin)
    fcolors2 = res2
    fcolors2 = (fcolors2 - fmin) / (fmax - fmin)
    fcolorsdiff = diff
    fcolorsdiff = (fcolorsdiff - fmin) / (fmax - fmin)

    ax1.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.plasma(fcolors1))
    ax2.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.plasma(fcolors2))
    ax3.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.plasma(fcolorsdiff))

    ax1.set_title("analytic")
    ax2.set_title("numeric")
    ax3.set_title("difference")
    # ax.grid()
    m = cm.ScalarMappable(cmap=cm.plasma)
    fig.colorbar(m, shrink=0.5, pad=0.15)
    plt.show()


def sphere_to_carthesian(r, theta, phi):
    return r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)


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
    db_v = 1 / 2 * (b1_v - b2_v)
    b1, b2, db, a = vnorm(b1_v), vnorm(b2_v), vnorm(db_v), vnorm(a_v)
    h = np.sqrt(a ** 2 - db ** 2)
    area = 1 / 2 * h * (b1 + b2)
    r_center = rs[0, 0, :] + b1_v / 2 + 1 / 2 * (a_v + db_v)
    n_v = np.cross(b1_v, a_v)
    return area, r_center, n_v


def kroen(a, b):
    if a == b:
        return 1
    else:
        return 0


def SCSM_trapezes(N=100, r=8, r0=np.array([0, 0, 11]), m=np.array([0, 1, 0]), sig=0.33, omega=1):
    eps0 = 8.854187812813e-12
    phis = np.linspace(0, 2 * np.pi, N)
    thetas = np.linspace(0, 2 * np.pi, N)
    M = N ** 2
    rs = np.zeros([M, 3])
    areas = np.zeros(M)
    rs_trap = np.zeros([2, 2, 3])  # 2x2 matrix with vectors as entries
    norm_vects = np.zeros([M, 3])
    A_real = np.zeros([M, M])
    A_imag = np.zeros([M, M])
    B = np.zeros(M)
    n = 0
    for i in range(N - 1):
        for j in range(N - 1):
            rs_trap[0, 0, :] = sphere_to_carthesian(r, thetas[i], phis[j])
            rs_trap[0, 1, :] = sphere_to_carthesian(r, thetas[i + 1], phis[j])
            rs_trap[1, 0, :] = sphere_to_carthesian(r, thetas[i], phis[j + 1])
            rs_trap[1, 1, :] = sphere_to_carthesian(r, thetas[i + 1], phis[j + 1])
            areas[n], rs[n, :], norm_vects[n, :] = trapezoid_area_and_centre(rs_trap)
            n += 1

    for u in range(M):
        for v in range(M):
            A1 = 1 / (4 * np.pi * eps0 * vnorm(rs[u, :] - rs[v, :]) ** 3 + kroen(u, v)) * (rs[u, :] - rs[v, :]) @ \
                 norm_vects[v]
            A2 = kroen(u, v) / 2 * eps0 * areas[u] * (1 / 2 + (omega * eps0 / sig) * 1j)
            A = np.array([A1, 0]) - np.array([A2.real, A2.imag])
            A_real[u, v] = A[0]
        B[u] = vnorm(1e-7 * (np.cross(m, (rs[u] - r0))) / (vnorm(rs[u] - r0) ** 3))

    Q = np.linalg.solve(A_real, B)
    return Q, rs


def SCSM_tri_sphere(tri_centers, areas, r0=np.array([0, 0, 1.1]), m=np.array([0, 1, 0]), sig=0.33, omega=1):
    rs = tri_centers
    M = rs.shape[0]
    A = np.zeros([M, M], dtype=np.complex_)
    B = np.zeros(M, dtype=np.complex_)
    eps0 = 8.854187812813e-12
    for i in range(M):
        r_norm_i = rs[i] / vnorm(rs[i])
        for j in range(M):
            r_norm_j = rs[j] / vnorm(rs[j])
            A[i, j] = np.dot((rs[i, :] - rs[j, :]), r_norm_i) / \
                      (4 * np.pi * eps0 * vnorm(rs[i, :] - rs[j, :]) ** 3 + kroen(i, j)) \
                      - kroen(i, j) / (eps0 * areas[i]) * ((1 / 2) + ((1j * omega * eps0) / sig))
        B[i] = 1j * omega * 1e-7 * np.dot(np.cross(m, (rs[i] - r0)), r_norm_i) / (vnorm(rs[i] - r0) ** 3)

    Q = np.linalg.solve(A, B)
    return Q, rs


@numba.jit(nopython=True, parallel=True)
def SCSM_tri_sphere_numba(tri_centers, tri_points, areas, r0=np.array([0, 0, 1.1]), m=np.array([0, 1, 0]), sig=0.33, omega=1):
    rs = tri_centers
    M = rs.shape[0]
    A = np.zeros((M, M), dtype=np.complex_)
    B = np.zeros(M, dtype=np.complex_)
    eps0 = 8.854187812813e-12

    def vnorm(x):
        return np.linalg.norm(x)

    def kroen(i, j):
        return int(i == j)

    for i in numba.prange(M):
        r_norm_i = rs[i] / vnorm(rs[i])
        p1 = tri_points[i][0]
        p2 = tri_points[i][1]
        p3 = tri_points[i][2]
        n = - np.cross((p3 - p1), (p2 - p3)) / (2 * areas[i])
        for j in numba.prange(M):
            A11 = np.dot((rs[i, :] - rs[j, :]), n)
            A12 = (4 * np.pi * eps0 * vnorm(rs[i, :] - rs[j, :]) ** 3 + kroen(i, j))
            A1 = A11 / A12
            A2 = kroen(i, j) / (eps0 * areas[i]) * ((1 / 2) + ((1j * omega * eps0) / sig))
            A[i, j] = A1 - A2
        B[i] = 1j * omega * 1e-7 * np.dot(np.cross(m, (rs[i] - r0)), n) / (vnorm(rs[i] - r0) ** 3)

    Q = np.linalg.solve(A, B)
    return Q, rs


def Q_parallel(idxs, A, B, rs, r0, m, areas, eps0, omega, sig, M):
    for k in range(len(idxs)):
        i = idxs[k][0]
        j = idxs[k][1]
        r_norm_i = rs[i] / vnorm(rs[i])
        A[i, j] = np.dot((rs[i, :] - rs[j, :]), r_norm_i) / \
                  (4 * np.pi * eps0 * vnorm(rs[i, :] - rs[j, :]) ** 3 + kroen(i, j)) \
                  - kroen(i, j) / (eps0 * areas[i]) * ((1 / 2) + ((1j * omega * eps0) / sig))
        if j == M - 1:
            B[i] = 1j * omega * 1e-7 * np.dot(np.cross(m, (rs[i] - r0)), r_norm_i) / (vnorm(rs[i] - r0) ** 3)


def A_parallel(idxs, A, rs, r_norm_i, areas, eps0, omega, sig, i):
    for k in range(len(idxs)):
        j = idxs[k]
        A[i, j] = np.dot((rs[i, :] - rs[j, :]), r_norm_i) / \
                  (4 * np.pi * eps0 * vnorm(rs[i, :] - rs[j, :]) ** 3 + kroen(i, j)) \
                  - kroen(i, j) / (eps0 * areas[i]) * ((1 / 2) + ((1j * omega * eps0) / sig))


def SCSM_Q_parallel(manager, tri_centers, areas, r0=np.array([0, 0, 1.1]), m=np.array([0, 1, 0]),
                    sig=0.33, omega=1):
    rs = tri_centers
    M = rs.shape[0]
    A = manager.np_zeros([M, M], dtype=np.complex_)
    B = manager.np_zeros(M, dtype=np.complex_)
    eps0 = 8.854187812813e-12
    n_cpu = multiprocessing.cpu_count()
    idx_sequence = product(list(range(M)), list(range(M)))
    idx_list = list(idx_sequence)
    pool = multiprocessing.Pool(n_cpu)
    workhorse_partial = partial(Q_parallel, A=A, B=B, rs=rs, r0=r0, m=m, areas=areas, eps0=eps0,
                                omega=omega, sig=sig, M=M)
    idx_list_chunks = compute_chunks(idx_list, n_cpu)
    pool.map(workhorse_partial, idx_list_chunks)
    pool.close()
    Q = np.linalg.solve(np.array(A), np.array(B))
    return Q, rs


@numba.jit(nopython=True, parallel=True)
def SCSM_E_sphere_numba(Q, r_q, r_sphere, theta, m=np.array([0, 1, 0]), r0=np.array([0, 0, 1.1]), omega=1,
                        phi=np.zeros((10, 10)), near_field=False, projection="polar"):
    eps0 = 8.854187812813e-12
    if projection == "polar":
        I = r_sphere.shape[0]
        J = theta.shape[0]
    elif projection == "sphere_surface":
        I = phi.shape[0]
        J = theta.shape[0]
    else:
        raise TypeError("projection can only be 'polar' or 'sphere_surface'!")
    N = r_q.shape[0]

    def vnorm(x):
        return np.linalg.norm(x)

    if projection == "polar":
        E = np.zeros((r_sphere.shape[0], theta.shape[0]))
        for i in numba.prange(I):
            for j in numba.prange(J):
                xs = r_sphere * np.cos(theta[j])
                ys = r_sphere * np.sin(theta[j])
                zs = np.zeros(xs.shape[0])
                rs = np.vstack((xs, ys, zs)).T
                r_v = rs[i]
                grad_phi = np.zeros((3, N), dtype=np.complex_)
                for n in numba.prange(N):
                    grad_phi[:, n] = Q[n] * (r_v - r_q[n]) / (4 * np.pi * eps0 * vnorm(r_v - r_q[n]) ** 3)
                E_complex = 1 * grad_phi.sum(axis=1) - 1 * (1j * omega * 1e-7) * (np.cross(m, (r_v - r0))) / (
                        vnorm(r_v - r0) ** 3)
                E[i, j] = vnorm(E_complex.imag)
    if projection == "sphere_surface":
        E = np.zeros((phi.shape[0], theta.shape[0]))
        for i in numba.prange(I):
            for j in numba.prange(J):
                xs = r_sphere * np.cos(phi[i]) * np.sin(theta[j])
                ys = r_sphere * np.sin(phi[i]) * np.sin(theta[j])
                zs = r_sphere * np.cos(theta[j])
                rs = np.vstack((xs, ys, zs)).T
                r_v = rs[i]
                grad_phi = np.zeros((3, N), dtype=np.complex_)
                for n in numba.prange(N):
                    grad_phi[:, n] = Q[n] * (r_v - r_q[n]) / (4 * np.pi * eps0 * vnorm(r_v - r_q[n]) ** 3)
                E_complex = 1 * grad_phi.sum(axis=1) - 1 * (1j * omega * 1e-7) * (np.cross(m, (r_v - r0))) / (
                        vnorm(r_v - r0) ** 3)
                E[i, j] = vnorm(E_complex.imag)

    return E

@numba.jit(nopython=True, parallel=True)
def SCSM_E_sphere_numba_polar(Q, r_q, r_sphere, theta, m=np.array([0, 1, 0]), r0=np.array([0, 0, 1.1]), omega=1,
                        phi=np.zeros((10, 10)), near_field=False, projection="polar"):
    eps0 = 8.854187812813e-12
    I = r_sphere.shape[0]
    J = theta.shape[0]
    N = r_q.shape[0]
    E = np.zeros((r_sphere.shape[0], theta.shape[0]))

    def vnorm(x):
        return np.linalg.norm(x)

    for i in numba.prange(I):
        for j in numba.prange(J):
            xs = r_sphere * np.cos(theta[j])
            ys = r_sphere * np.sin(theta[j])
            zs = np.zeros(xs.shape[0])
            rs = np.vstack((xs, ys, zs)).T
            r_v = rs[i]
            grad_phi = np.zeros((3, N), dtype=np.complex_)
            for n in numba.prange(N):
                grad_phi[:, n] = Q[n] * (r_v - r_q[n]) / (4 * np.pi * eps0 * vnorm(r_v - r_q[n]) ** 3)
            E_complex = 1 * grad_phi.sum(axis=1) - 1 * (1j * omega * 1e-7) * (np.cross(m, (r_v - r0))) / (
                    vnorm(r_v - r0) ** 3)
            E[i, j] = vnorm(E_complex.imag)

    return E

@numba.jit(nopython=True, parallel=True)
def SCSM_E_sphere_numba_surf(Q, r_q, r_sphere, theta, m=np.array([0, 1, 0]), r0=np.array([0, 0, 1.1]), omega=1,
                        phi=np.zeros((10, 10)), near_field=False):
    eps0 = 8.854187812813e-12
    I = phi.shape[0]
    J = theta.shape[0]
    N = r_q.shape[0]
    E = np.zeros((phi.shape[0], theta.shape[0]))

    def vnorm(x):
        return np.linalg.norm(x)

    for i in numba.prange(I):
        # phi_i = phi[i]
        for j in numba.prange(J):
            phi_i = phi.T[0][i]
            theta_j = theta[0][j]
            x = r_sphere * np.sin(phi_i) * np.cos(theta_j)
            y = r_sphere * np.sin(phi_i) * np.sin(theta_j)
            z = r_sphere * np.cos(phi_i)
            r_v = np.array((x, y, z))
            grad_phi = np.zeros((3, N), dtype=np.complex_)
            for n in numba.prange(N):
                grad_phi[:, n] = Q[n] * (r_v - r_q[n]) / (4 * np.pi * eps0 * vnorm(r_v - r_q[n]) ** 3)
            E_complex = 1 * grad_phi.sum(axis=1) - 1 * (1j * omega * 1e-7) * (np.cross(m, (r_v - r0))) / (
                    vnorm(r_v - r0) ** 3)
            E[i, j] = vnorm(E_complex.imag)
    return E


def SCSM_E_sphere(Q, r_q, r_sphere, theta, m=np.array([0, 1, 0]), r0=np.array([0, 0, 1.1]), omega=1):
    eps0 = 8.854187812813e-12
    I, J, N = r_sphere.shape[0], theta.shape[0], r_q.shape[0]
    E = np.zeros([r_sphere.shape[0], theta.shape[0]])
    start_time = time.time()
    time_1 = 0
    for i in range(I):
        for j in range(J):

            xs, ys = r_sphere * np.cos(theta[j]), r_sphere * np.sin(theta[j])
            rs = np.array([xs, ys, np.zeros(xs.shape[0])])
            r_v = rs[:, i]
            grad_phi = np.zeros([3, N], dtype=np.complex_)
            for n in range(N):
                grad_phi[:, n] = Q[n] * (r_v - r_q[n]) / (4 * np.pi * eps0 * vnorm(r_v - r_q[n]) ** 3)
            E_complex = grad_phi.sum(axis=1) - (1j * omega * 4 * np.pi * 1e-7) / (4 * np.pi * vnorm(r_v - r0) ** 3) * (
                np.cross(m, (r_v - r0)))
            # E[i, j] = vnorm(E_complex.imag)
            # E[i, j, 0, :] = E_complex.real
            E[i, j] = vnorm(E_complex.imag)
        if i == 0:
            time_1 = time.time()
        print(f"radius {i + 1} of {theta.shape[0]} done, remaining time: "
              f"{format(((time_1 - start_time) * theta.shape[0] - (time.time() - start_time)) / 60, '.1f')} minutes")

    return E


def E_parallel(idxs, E, Q, r_sphere, r_q, r0, theta, m, phi, omega, eps0, mu0, N, projection, near_field, tri_points,
               near_radius):
    for k in range(len(idxs)):
        i = idxs[k][0]
        j = idxs[k][1]
        if projection == "polar":
            xs, ys = r_sphere * np.cos(theta[j]), r_sphere * np.sin(theta[j])
            rs = np.array([xs, ys, np.zeros(xs.shape[0])])
            r_v = rs[:, i]
        elif projection == "sphere_surface":
            phi_i = phi[i, 0]
            theta_j = theta[0, j]
            x, y, z = r_sphere * np.sin(phi_i) * np.cos(theta_j), r_sphere * np.sin(phi_i) * np.sin(theta_j), \
                      r_sphere * np.cos(phi_i)
            r_v = np.array([x, y, z])
        else:
            raise TypeError("projection can only be 'polar' or 'sphere_surface'!")

        grad_phi = np.zeros([3, N], dtype=np.complex_)
        for n in range(N):
            if near_field:
                # eps_r = vnorm(r_q[n] - r_v)
                # if eps_r < near_radius:
                grad_phi[:, n] = E_near(Q[n], tri_points[n][0], tri_points[n][1], tri_points[n][2], r_v)
                # else:
                #     grad_phi[:, n] = Q[n] * (r_v - r_q[n]) / (4 * np.pi * eps0 * vnorm(r_v - r_q[n]) ** 3)
            else:
                grad_phi[:, n] = Q[n] * (r_v - r_q[n]) / (4 * np.pi * eps0 * vnorm(r_v - r_q[n]) ** 3)
        E_complex = 1 * grad_phi.sum(axis=1) - 1 * (1j * omega * 1e-7) * (np.cross(m, (r_v - r0))) / (
                    vnorm(r_v - r0) ** 3)
        E[i, j] = vnorm(E_complex.imag)


def parallel_SCSM_E_sphere(manager, Q, r_q, r_sphere, theta, m=np.array([0, 1, 0]), r0=np.array([0, 0, 1.1]),
                           projection="polar", phi=np.zeros((10, 10)), omega=1, near_field=False, tri_points=None,
                           near_radius=0.1):
    eps0 = 8.854187812813e-12
    mu0 = 4 * np.pi * 1e-7
    # E_v = np.zeros([3, r_q.shape[0]], dtype=np.complex_)

    if near_field and tri_points is None:
        raise ValueError("near_field approximation specified, but no valid triangle points were given!")

    # manager.start()
    if projection == "polar":
        I, J, N = r_sphere.shape[0], theta.shape[0], r_q.shape[0]
    elif projection == "sphere_surface":
        I, J, N = phi.shape[0], theta.shape[0], r_q.shape[0]
    else:
        raise TypeError("projection can only be 'polar' or 'sphere_surface'!")
    E = manager.np_zeros([I, J])
    # n_cpu = 5
    n_cpu = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(n_cpu)
    idx_sequence = product(list(range(I)), list(range(J)))
    idx_list = list(idx_sequence)

    workhorse_partial = partial(E_parallel, E=E, Q=Q, r_sphere=r_sphere, r_q=r_q, r0=r0, theta=theta,
                                m=m, phi=phi, omega=omega, eps0=eps0, mu0=mu0, N=N, projection=projection,
                                near_field=near_field, near_radius=near_radius, tri_points=tri_points)
    # for i in range(I):
    #     for j in range(J):
    #
    #         if projection == "polar":
    #             xs, ys = r_sphere * np.cos(theta[j]), r_sphere * np.sin(theta[j])
    #             rs = np.array([xs, ys, np.zeros(xs.shape[0])])
    #             r_v = rs[:, i]
    #         elif projection == "sphere_surface":
    #             x, y, z = r_sphere * np.cos(phi[i]) * np.sin(theta[j]), r_sphere * np.sin(phi[i]) * np.sin(theta[j]),\
    #                       r_sphere * np.cos(theta[j])
    #             r_v = np.array([x, y, z])
    #         else:
    #             raise TypeError("projection can only be 'polar' or 'sphere_surface'!")
    #
    #         grad_phi = np.zeros([3, N], dtype=np.complex_)
    #         for n in range(N):
    #             if near_field:
    #                 if near_field:
    #                     # eps_r = vnorm(r_q[n] - r_v)
    #                     # if eps_r < near_radius:
    #                     grad_phi[:, n] = E_near(Q[n], tri_points[n][0], tri_points[n][1], tri_points[n][2], r_v)
    #             else:
    #                 grad_phi[:, n] = Q[n] * (r_v - r_q[n]) / (4 * np.pi * eps0 * vnorm(r_v - r_q[n]) ** 3)
    #         E_complex = 1 * grad_phi.sum(axis=1) - 1 * (1j * omega * 1e-7) * (np.cross(m, (r_v - r0))) / (
    #                     vnorm(r_v - r0) ** 3)
    #         E[i, j] = vnorm(E_complex.imag)

    # for i in range(I):
    #     for j in range(J):
    #
    #         xs = r_sphere * np.cos(theta[j])
    #         ys = r_sphere * np.sin(theta[j])
    #         zs = np.zeros(xs.shape[0])
    #         rs = np.vstack((xs, ys, zs)).T
    #         r_v = rs[i]
    #         grad_phi = np.zeros((3, N), dtype=np.complex_)
    #         for n in range(N):
    #             grad_phi[:, n] = Q[n] * (r_v - r_q[n]) / (4 * np.pi * eps0 * vnorm(r_v - r_q[n]) ** 3)
    #         E_complex = grad_phi.sum(axis=1) - (1j * omega * 4 * np.pi * 1e-7) / (4 * np.pi * vnorm(r_v - r0) ** 3) * (
    #             np.cross(m, (r_v - r0)))
    #         E[i, j] = vnorm(E_complex.imag)

    idx_list_chunks = compute_chunks(idx_list, n_cpu)
    pool.map(workhorse_partial, idx_list_chunks)
    pool.close()

    return np.array(E)


# r = 8
# theta, phi  = np.meshgrid(np.linspace(0, 2*np.pi, 100), np.linspace(0, 2*np.pi, 100))
# fig = plt.figure()
# xs, ys, zs = r*np.cos(phi)*np.sin(theta),  r*np.sin(phi)*np.sin(theta),  r*np.cos(theta)
# ax = plt.subplot(111, projection='3d')
# ax.scatter(xs, ys, zs)
# plt.show()

def E_near(Q, p1, p2, p3, r):
    """
    Calculates the electric field norm of one charged triangular sheet
    :param Q: Charge of the triangle
    :param p1: point 1 of the triangle
    :param p2: point 2 of the triangle
    :param p3: point 3 of the triangle
    :param r: vector where the electric field is evaluated
    :return E: electric field norm at r
    """
    eps0 = 8.854187812813e-12
    c = (1 / 3) * (p1 + p2 + p3)
    A = np.linalg.norm(np.cross((p3 - p1), (p2 - p1))) / 2
    n = np.cross((p3 - p1), (p2 - p3)) / (2 * A)
    h = np.dot(n, (r - c))
    p0 = r - (h * n)
    r1, r2, r3 = np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p0), np.linalg.norm(p3 - p0)
    d1, d2, d3 = np.linalg.norm(p1 - r), np.linalg.norm(p2 - r), np.linalg.norm(p3 - r)
    s12, s23, s31 = np.linalg.norm(p2 - p1), np.linalg.norm(p3 - p2), np.linalg.norm(p1 - p3)
    D12, D23, D31 = np.dot(p1 - p0, p2 - p0), np.dot(p2 - p0, p3 - p0), np.dot(p3 - p0, p1 - p0)
    c12, c23, c31 = np.dot(n, np.cross((p2 - p0), (p1 - p0))), np.dot(n, np.cross((p3 - p0), (p2 - p0))), \
                    np.dot(n, np.cross((p1 - p0), (p3 - p0)))
    D1 = h ** 2 * (r1 ** 2 + D23 - D12 - D31) - (c12 * c31)
    D2 = h ** 2 * (r2 ** 2 + D31 - D23 - D12) - (c23 * c12)
    D3 = h ** 2 * (r3 ** 2 + D12 - D31 - D23) - (c31 * c23)
    N = -h * (c12 + c23 + c31)

    k12x = np.array((0, p1[2] - p2[2], p2[1] - p1[1]))
    k23x = np.array((0, p2[2] - p3[2], p3[1] - p2[1]))
    k31x = np.array((0, p3[2] - p1[2], p1[1] - p3[1]))
    k12y = np.array((p2[2] - p1[2], 0, p1[0] - p2[0]))
    k23y = np.array((p3[2] - p2[2], 0, p2[0] - p3[0]))
    k31y = np.array((p1[2] - p3[2], 0, p3[0] - p1[0]))
    k12z = np.array((p1[1] - p2[1], p2[0] - p1[0], 0))
    k23z = np.array((p2[1] - p3[1], p3[0] - p2[0], 0))
    k31z = np.array((p3[1] - p1[1], p1[0] - p3[0], 0))
    # f for x component
    if s12 == 0 or s12 == d1 + d2:
        f12x = 0
    else:
        f12x = np.dot(n, k12x)/s12*np.log((d1 + d2 + s12)/(d1 + d2 - s12))
    if s23 == 0 or s23 == d2 + d3:
        f23x = 0
    else:
        f23x = np.dot(n, k23x)/s23*np.log((d2 + d3 + s23)/(d2 + d3 - s23))
    if s31 == 0 or s31 == d3 + d1:
        f31x = 0
    else:
        f31x = np.dot(n, k31x)/s31*np.log((d3 + d1 + s31)/(d3 + d1 - s31))
    # f for y component
    if s12 == 0 or s12 == d1 + d2:
        f12y = 0
    else:
        f12y = np.dot(n, k12y)/s12*np.log((d1 + d2 + s12)/(d1 + d2 - s12))
    if s23 == 0 or s23 == d2 + d3:
        f23y = 0
    else:
        f23y = np.dot(n, k23y)/s23*np.log((d2 + d3 + s23)/(d2 + d3 - s23))
    if s31 == 0 or s31 == d3 + d1:
        f31y = 0
    else:
        f31y = np.dot(n, k31y)/s31*np.log((d3 + d1 + s31)/(d3 + d1 - s31))
    # f for z component
    if s12 == 0 or s12 == d1 + d2:
        f12z = 0
    else:
        f12z = np.dot(n, k12z)/s12*np.log((d1 + d2 + s12)/(d1 + d2 - s12))
    if s23 == 0 or s23 == d2 + d3:
        f23z = 0
    else:
        f23z = np.dot(n, k23z)/s23*np.log((d2 + d3 + s23)/(d2 + d3 - s23))
    if s31 == 0 or s31 == d3 + d1:
        f31z = 0
    else:
        f31z = np.dot(n, k31z)/s31*np.log((d3 + d1 + s31)/(d3 + d1 - s31))
    # perpendicular components
    arc_tangent_sum = np.arctan2(D1, N * d1) + np.arctan2(D2, N * d2) + np.arctan2(D3, N * d3) + np.sign(h) * np.pi
    arc_tangent_sum = np.arctan(D1 / (N * d1)) + np.arctan(D2 / (N * d2)) + np.arctan(D3 / (N * d3)) + np.sign(h) * np.pi/2
    gx = - n[0] * arc_tangent_sum
    gy = - n[1] * arc_tangent_sum
    gz = - n[2] * arc_tangent_sum

    Ex = f12x + f23x + f31x + gx
    Ey = f12y + f23y + f31y + gy
    Ez = f12z + f23z + f31z + gz

    E = -Q/(A * 4*np.pi * eps0) * np.array((Ex, Ey, Ez))
    return E


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


def fibonacci_sphere_mesh(samples=1000):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points
