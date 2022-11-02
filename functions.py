from typing import Union, Any

import numpy as np
from scipy.special import legendre
from itertools import product
from functools import partial
from mpl_toolkits import mplot3d
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
import fmm3dpy as fmm
import cupy as cp
from numba import cuda
# import cupyx
# from mpmath import fp
# import dask.array as da


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


def read_sphere_mesh_from_txt_locations_only(sizes, path, scaling=1):
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

    return triangle_centers, areas, triangle_points

def sphere_mesh(samples=1000, scaling=1):

    locations = scaling * fibonacci_sphere_mesh(samples=samples).T
    sphere_surf_locations = carthesian_to_sphere(locations.T)[:, 1:]
    tri = Delaunay(sphere_surf_locations)
    connections = tri.simplices.copy().T

    triangle_centers = np.zeros([len(connections[0, :]), 3])
    areas = np.zeros(len(connections[0, :]))
    n_v = np.zeros_like(triangle_centers)


    n = len(connections[0, :])
    triangle_points = np.zeros((n, 3, 3))
    edge_lens = np.zeros(n)
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
        line2_3 = p3 - p2
        areas[i] = 0.5 * vnorm(np.cross(line1_2, line1_3))
        n_v[i] = - np.cross((p3 - p1), (p2 - p3)) / (2 * areas[i])
        edge_lens[i] = 1/3*(np.linalg.norm(line1_2) + np.linalg.norm(line1_3) + np.linalg.norm(line2_3))
    avg_length = np.mean(edge_lens)
    return triangle_centers, areas, triangle_points, n_v, avg_length

def read_mesh_from_hdf5(fn, mode="source"):

    if mode == "source":
        with h5py.File(fn, "r") as f:
            mesh_data = f['mesh']
            elm_number = np.array(f['mesh']['elm']['elm_number'])
            elm_type = np.array(f['mesh']['elm']['elm_type'])
            node_number_list = np.array(f['mesh']['elm']['node_number_list'])
            tri_tissue_type = np.array(f['mesh']['elm']['tri_tissue_type'])
            triangle_number_list = np.array(f['mesh']['elm']['triangle_number_list'])
            node_number = np.array(f['mesh']['nodes']['node_number'])
            node_coords = np.array(f['mesh']['nodes']['node_coord'])
        # don't consider eyeballs and cerebellum for evaluation (tissue_type 1006, 1ßß7)
        # tris_wm = triangle_number_list[np.where(tri_tissue_type == 1001)]
        # tris_gm = triangle_number_list[np.where(tri_tissue_type == 1002)]
        # tris_csf = triangle_number_list[np.where(tri_tissue_type == 1003)]
        # tris_skull = triangle_number_list[np.where(tri_tissue_type == 1004)]
        # tris_skin = triangle_number_list[np.where(tri_tissue_type == 1005)]
        # tris = np.vstack([tris_wm, tris_gm, tris_csf, tris_skull, tris_skin])
        tri_tissue_type = tri_tissue_type[np.where(tri_tissue_type < 1006)]
        points = triangle_number_list[np.where(tri_tissue_type < 1006)]

        n = points.shape[0]
        triangle_centers = np.zeros([n, 3])
        areas = np.zeros(n)
        n_v = np.zeros_like(triangle_centers)

        triangle_points = np.zeros((n, 3, 3))
        for i in range(n):
            p1 = node_coords[int(points[i, 0]), :]
            p2 = node_coords[int(points[i, 1]), :]
            p3 = node_coords[int(points[i, 2]), :]
            triangle_points[i][:][:] = np.vstack((p1, p2, p3))
            p_c_1 = (p1[0] + p2[0] + p3[0]) / 3
            p_c_2 = (p1[1] + p2[1] + p3[1]) / 3
            p_c_3 = (p1[2] + p2[2] + p3[2]) / 3
            triangle_centers[i, :] = np.array([p_c_1, p_c_2, p_c_3])
            line1_2 = p2 - p1
            line1_3 = p3 - p1
            areas[i] = 0.5 * np.linalg.norm(np.cross(line1_2, line1_3))
            n_v[i] = - np.cross((p3 - p1), (p2 - p3)) / (2 * areas[i])
        return triangle_centers, areas, triangle_points, n_v, tri_tissue_type

    elif mode == "target":
        with h5py.File(fn, "r") as f:
            triangle_centers = np.array(f['roi_surface']['midlayer_m1s1pmd']['tri_center_coord_mid'])
        return triangle_centers

    elif mode == "coil":
        with h5py.File(fn, "r") as f:
            transformation_matrix = np.array(f['info']['matsimnibs'])
            sigmas = np.array((np.array(f['info']['sigma_WM']), np.array(f['info']['sigma_GM']),
                               np.array(f['info']['sigma_CSF']), np.array(f['info']['sigma_Skull']),
                               np.array(f['info']['sigma_Scalp']), 0))
        return transformation_matrix, sigmas
    else:
        raise ValueError("mode can only be 'source', 'target' or 'coil'!")

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
                x, y, z = r_sphere * np.sin(phi_i) * np.cos(theta_j), r_sphere * np.sin(phi_i) * np.sin(theta_j), \
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

def reciprocity_sphere(grid, n=100, r0_v=np.array([12, 0, 0]), m=np.array([0, -1, 0]), omega=1):
    mu0 = 4 * np.pi * 1e-7
    r0 = vnorm(r0_v)

    E = np.zeros([n, n])
    Ev = np.zeros([n, n, 3])
    for i in range(n):
        for j in range(n):
            r_v = np.array([grid[0][i, j], grid[1][i, j], grid[2][i, j]])
            a_v = r0_v - r_v
            a = vnorm(a_v)
            F = (r0 * a + np.dot(r0_v, a_v)) * a
            nab_F = ((a ** 2 / r0 ** 2) + 2 * a + 2 * r0 + (np.dot(r0_v, a_v) / a)) * r0_v - (
                    a + 2 * r0 + (np.dot(r0_v, a_v) / a)) * r_v
            E_v = omega * mu0 / (4 * np.pi * F ** 2) * (
                        F * np.cross(r_v, m) - np.dot(m, nab_F) * np.cross(r_v, r0_v))
            Ev[i, j, :] = E_v
            E[i, j] = vnorm(E_v)
    return Ev, E

# @numba.jit(parallel=True)
def reciprocity_surface(rs, r0_v=np.array([12, 0, 0]), m=np.array([0, -1, 0]), omega=1):
    M = rs.shape[0]
    mu0 = 4 * np.pi * 1e-7
    E = np.zeros(M)
    E_v = np.zeros((M, 3))

    def pdot(a, b):
        return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]

    def pfact(f, array):
        return np.vstack((f[:] * array[:, 0], f[:] * array[:, 1],  f[:] * array[:, 2])).T

    for i in numba.prange(M):
        r_v = rs[i]
        r0 = np.linalg.norm(r0_v, axis=1)
        r_v_long = np.tile(r_v, (m.shape[0], 1))
        a_v = r0_v - r_v
        a = np.linalg.norm(a_v, axis=1)
        F = (r0 * a + pdot(r0_v, a_v)) * a
        nab_F = pfact(((a ** 2 / r0 ** 2) + 2 * a + 2 * r0 + (pdot(r0_v, a_v) / a)), r0_v) - pfact((
                a + 2 * r0 + (pdot(r0_v, a_v) / a)), r_v_long)
        E_vi = pfact(omega * mu0 / (4 * np.pi * F ** 2), (pfact(F, np.cross(r_v_long, m)) -
                                                           pfact(pdot(m, nab_F), np.cross(r_v_long, r0_v))))
        E_v[i, :] = np.sum(E_vi, axis=0)
        # E[i] = np.sum(np.linalg.norm(E_v, axis=1))
        E[i] = np.linalg.norm(np.sum(E_vi.T, axis=1))
    return E_v, E

def reciprocity_surface_single_m(rs, r0_v=np.array([12, 0, 0]), m=np.array([0, -1, 0]), omega=1):
    M = rs.shape[0]
    mu0 = 4 * np.pi * 1e-7
    E = np.zeros(M)
    E_v = np.zeros((M, 3))

    for i in numba.prange(M):
        r_v = rs[i]
        r0 = np.linalg.norm(r0_v)
        r_v_long = r_v
        a_v = r0_v - r_v
        a = np.linalg.norm(a_v)
        F = (r0 * a + np.dot(r0_v, a_v)) * a
        nab_F = ((a ** 2 / r0 ** 2) + 2 * a + 2 * r0 + (np.dot(r0_v, a_v) / a)) * r0_v - (
                a + 2 * r0 + (np.dot(r0_v, a_v) / a)) * r_v_long
        E_vi = omega * mu0 / (4 * np.pi * F ** 2) * (F * np.cross(r_v_long, m) -
                                                     np.dot(m, nab_F) * np.cross(r_v_long, r0_v))
        E_v[i, :] = E_vi
        # E[i] = np.sum(np.linalg.norm(E_v, axis=1))
        E[i] = np.linalg.norm(E_vi)
    return E_v, E



@numba.njit(parallel=True)
def sphere_to_carth_numba(r_sphere, theta, phi):
    for i_phi in numba.prange(phi.shape[0]):
        for i_theta in numba.prange(theta.shape[0]):
            phi_i = phi[i_phi, 0]
            theta_j = theta[0, i_theta]
            x, y, z = r_sphere * np.sin(phi_i) * np.cos(theta_j), r_sphere * np.sin(phi_i) * np.sin(theta_j), \
                      r_sphere * np.cos(phi_i)
            r = np.array([x, y, z])
    return r


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


def plot_E_diff_test(res1, res2, r, theta, r_max, r0=None, m=None):
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

def plot_E_diff(res1, res2, r, theta, r_max, r0=None, m=None):
    diff = np.abs(res2 - res1)
    rerror = np.linalg.norm(diff) * 100 / np.linalg.norm(res1)
    res1 = (res1 - res1.min()) / (res1.max() - res1.min())
    res2 = (res2 - res2.min()) / (res2.max() - res2.min())
    diff = (diff - diff.min()) / (diff.max() - diff.min())
    fig, ax = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=(14, 5))
    ax1, ax2, ax3 = ax[0], ax[1], ax[2]
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
    fig.colorbar(im)
    f_maxd = diff.max()
    f_mind = diff.min()
    im = ax3.pcolormesh(theta, r, diff, cmap='plasma', vmin=f_mind, vmax=f_maxd)
    ax3.set_yticklabels([])
    ax3.set_ylim(0, r_max)
    # ax3.set_yticks(np.arange(0, r_max, r_max / 10))
    ax3.grid(True)

    ax1.set_title("analytic")
    ax2.set_title("numeric")
    ax3.set_title("difference")
    plt.subplots_adjust(wspace=0.7)
    plt.subplots_adjust(hspace=0.5)

    fig.suptitle(f"Electric field magnitudes and difference normalized by their respective maximum \n relative error: "
                 f"{rerror:.4f} %")
    plt.show()


def plot_E_sphere_surf(res, phi, theta, r, c_map=cm.plasma):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    fcolors = res
    fmax, fmin = fcolors.max(), fcolors.min()
    fcolors = (fcolors - fmin) / (fmax - fmin)
    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=c_map(fcolors))
    ax.grid()
    m = cm.ScalarMappable(cmap=c_map)
    fig.colorbar(m, shrink=0.5, pad=0.15)
    plt.show()

def plot_E_sphere_surf_diff(res1, res2, phi=None, theta=None, r=None, xyz_grid=None, c_map=cm.plasma, normalize=True,
                            names=["analytic", "numeric"], save=False, save_fn=None, plot_difference=True):
    diff = np.abs(res2 - res1)
    rerror = nrmse(res2, res1) * 100
    if plot_difference:
        fig, ax = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(14, 5))
        ax1, ax2, ax3 = ax[0], ax[1], ax[2]
    else:
        fig, ax = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(14, 5))
        ax1, ax2 = ax[0], ax[1]
    if xyz_grid is None:
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
    else:
        x = xyz_grid[0]
        y = xyz_grid[1]
        z = xyz_grid[2]

    if normalize:
        fcolors1 = res1
        fmax1, fmin1 = fcolors1.max(), fcolors1.min()
        fcolors1 = (fcolors1 - fmin1) / (fmax1 - fmin1)
        fcolors2 = res2
        fmax2, fmin2 = fcolors2.max(), fcolors2.min()
        fcolors2 = (fcolors2 - fmin1) / (fmax1 - fmin1)
        fcolorsdiff = diff
        fmax_d, fmin_d = fcolorsdiff.max(), fcolorsdiff.min()
        fcolorsdiff = (fcolorsdiff - fmin1) / (fmax1 - fmin1)
        min_val1, min_val2, min_vald = 0, 0, 0
        max_val1, max_val2, max_vald = 1, 1, 1
    else:
        fcolors1 = res1
        min_val1 = res1.min()
        max_val1 = res1.max()
        fcolors2 = res2
        min_val2 = res2.min()
        max_val2 = res2.max()
        fcolorsdiff = diff
        min_vald = diff.min()
        max_vald = diff.max()

    ax1.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=c_map(fcolors1))
    ax2.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=c_map(fcolors2))

    ax1.set_title(names[0])
    ax2.set_title(names[1])

    if plot_difference:
        ax3.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=c_map(fcolorsdiff))
        ax3.set_title("difference")

    # ax.grid()
    m = cm.ScalarMappable(cmap=c_map)
    cbar = fig.colorbar(m, shrink=0.5, pad=0.15)
    # cbar.set_label("normalized difference", rotation=270)
    # rerror = np.linalg.norm(diff) * 100 / np.linalg.norm(res1)
    if plot_difference:
        diff_str = "and difference"
    else:
        diff_str = " "
    fig.suptitle(f"normalized Electric fields " + diff_str + f" \n nrmse: {rerror:.4f} %")
    if not save:
        plt.show()
    else:
        plt.savefig(save_fn)
        plt.close()




def sphere_to_carthesian(r, theta, phi):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.vstack((x, y, z)).T

def circle_to_carthesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros(r.shape[0])
    return np.vstack((x, y, z)).T


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


def SCSM_tri_sphere(tri_centers, tri_points, areas, r0=np.array([0, 0, 1.1]), m=np.array([0, 1, 0]), sig=0.33, omega=1):
    rs = tri_centers
    M = rs.shape[0]
    A = np.zeros((M, M), dtype=np.complex_)
    B = np.zeros(M, dtype=np.complex_)
    eps0 = 8.854187812813e-12

    def vnorm(x):
        return np.linalg.norm(x)

    def kroen(i, j):
        return int(i == j)

    for i in range(M):
        p1 = tri_points[i][0]
        p2 = tri_points[i][1]
        p3 = tri_points[i][2]
        n = - np.cross((p3 - p1), (p2 - p3)) / (2 * areas[i])
        for j in range(M):
            A11 = np.dot((rs[i, :] - rs[j, :]), n)
            A12 = (4 * np.pi * eps0 * vnorm(rs[i, :] - rs[j, :]) ** 3 + kroen(i, j))
            A1 = A11 / A12
            A2 = kroen(i, j) / (eps0 * areas[i]) * ((1 / 2) + ((1j * omega * eps0) / sig))
            A[i, j] = A1 - A2
        B[i] = 1j * omega * 1e-7 * np.dot(np.cross(m, (rs[i] - r0)), n) / (vnorm(rs[i] - r0) ** 3)
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
    # B = 1j * omega * 1e-7 * np.dot(np.cross(m, (rs - r0)), (rs / vnorm(rs))) / (vnorm(rs - r0) ** 3)
    Q = np.linalg.solve(A, B)

    return Q, rs

@numba.jit(nopython=True, parallel=True)
def SCSM_matrix(tri_centers, areas, n, b_im=0, sig_in=0.33, sig_out=0.0, omega=1):
    rs = tri_centers
    M = rs.shape[0]
    a = np.zeros((M, M), dtype=np.complex_)
    eps0 = 8.854187812813e-12

    def vnorm(x):
        return np.linalg.norm(x)

    def kroen(i, j):
        return int(i == j)

    def v_vnorm(x):
        x = x.T
        return np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2).T
    f = 4 * np.pi * eps0

    d = - (((1 / 2) * (sig_in + sig_out) / (sig_in - sig_out)) + ((1j * omega * eps0) / (sig_in - sig_out))) * (
            1 / eps0 / areas)
    for i in numba.prange(M):

        # delta_r = rs[i] - rs
        # A11 = (delta_r[:, 0] * n[i, 0]) + (delta_r[:, 1] * n[i, 1]) + (delta_r[:, 2] * n[i, 1])
        # A12 = f * v_vnorm(delta_r) ** 3
        # a_i = A11 / (A12 + 1e-20)
        for j in numba.prange(M):
            a[i, j] = (np.dot((rs[i, :] - rs[j, :]), n[i])) / ((4 * np.pi * eps0 * vnorm(rs[i, :] - rs[j, :]) ** 3 + kroen(i, j)))
    B = 1j * b_im
    A = a + np.diag(d)
    # B = 1j * omega * 1e-7 * np.dot(np.cross(m, (rs - r0)), (rs / vnorm(rs))) / (vnorm(rs - r0) ** 3)
    Q = np.linalg.solve(A, B)
    return Q

@numba.jit(nopython=True, parallel=True)
def SCSM_jacobi_iter(tri_centers, tri_points, areas, n, r0=np.array([0, 0, 1.1]), m=np.array([0, 1, 0]), sig=0.33,
                          omega=1, n_iter=1000, tol=1e-15, A=None, b=None, initial_guess=False, b_im=None,
                     calc_b=True, verbose=False):

    rs = tri_centers
    M = rs.shape[0]
    eps0 = 8.854187812813e-12

    def vnorm(x):
        return np.linalg.norm(x)

    def v_vnorm(x):
        x = x.T
        return np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2).T

    x = np.zeros(M, np.complex_)
    a_ii = - ((1 / 2) + ((1j * omega * eps0) / sig)) / eps0 / areas
    f = 4 * np.pi * eps0
    if initial_guess:
        x = 1j * b_im / a_ii
    for it_count in numba.prange(n_iter):
        # if it_count != 0 and verbose:
        #     print("Iteration {0}: {1}".format(it_count, x))
        x_new = np.zeros_like(x)
        for i in numba.prange(M):
            if not A is None and not b is None:
                a_i = A[i, :]
                b_i = b[i]
            else:
                delta_r = rs[i] - rs
                A11 = (delta_r[:, 0] * n[i, 0]) + (delta_r[:, 1] * n[i, 1]) + (delta_r[:, 2] * n[i, 1])
                A12 = f * v_vnorm(delta_r) ** 3
                a_i = A11 / A12 + 1e-25j
                b_i = 1j * b_im[i]
            s1 = np.dot(a_i[:i], x[:i])
            s2 = np.dot(a_i[i + 1:], x[i + 1:])
            x_new[i] = (b_i - s1 - s2) / a_ii[i]
            if x_new[i] == x_new[i - 1]:
                break
        if vnorm(x - x_new) < tol:
            break
        x = x_new
    return x

def SCSM_jacobi_iter_cupy(tri_centers, areas, n, b_im, sig_in=0.33, sig_out=0.0, omega=3e3, n_iter=1000, tol=1e-15,
                          high_precision=False, verbose=False):

    if high_precision:
        rs = cp.asarray(tri_centers, dtype='float64')
        areas = cp.asarray(areas, dtype='float64')
        n = cp.asarray(n, dtype='float64')
        b_im = cp.asarray(b_im, dtype='float64')
        sig_out = cp.asarray(sig_out, dtype='float64')
        sig_in = cp.asarray(sig_in, dtype='float64')
    else:
        rs = cp.asarray(tri_centers, dtype='float32')
        areas = cp.asarray(areas, dtype='float32')
        n = cp.asarray(n, dtype='float32')
        b_im = cp.asarray(b_im, dtype='float32')
        sig_out = cp.asarray(sig_out, dtype='float32')
        sig_in = cp.asarray(sig_in, dtype='float32')

    M = rs.shape[0]
    eps0 = 8.854187812813e-12

    def vnorm(x):
        return cp.linalg.norm(x)

    def v_vnorm(x):
        x = x.T
        return cp.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2).T

    a_ii = - ( ((1/2) * (sig_in + sig_out) / (sig_in - sig_out)) + ((1j * omega * eps0) / (sig_in - sig_out)) ) * (1 / eps0 / areas)
    # a_ii = - ((1 / 2) + ((1j * omega * eps0) / 0.33)) * (1 / eps0 / areas)
    # if high_precision:
    #     x = cp.zeros(M, dtype='complex128')
    # else:
    #     x = cp.zeros(M, dtype='complex64')
    x = (1j * b_im) / a_ii
    f = 4 * cp.pi * eps0
    for i_iter in numba.prange(n_iter):
        x_new = cp.zeros_like(x)
        for i in numba.prange(M):
            delta_r = rs[i] - rs
            A11 = (delta_r[:, 0] * n[i, 0]) + (delta_r[:, 1] * n[i, 1]) + (delta_r[:, 2] * n[i, 1])
            A12 = f * v_vnorm(delta_r) ** 3
            a_i = A11 / (A12 + 1e-20)
            b_i = 1j * b_im[i]
            s1 = cp.dot(a_i[:i], x[:i])
            s2 = cp.dot(a_i[i + 1:], x[i + 1:])
            x_new[i] = (b_i - s1 - s2) / a_ii[i]

        if vnorm(x - x_new) < tol:
            if verbose:
                print(f"jacobi converged after {i_iter + 1} iterations with error norm {vnorm(x - x_new):.2g}")
            break
        else:
            if verbose:
                print(f"x_new = {x_new[:2]}")
                # if i_iter == 0:
                # print(f"a_ii[-1] = {a_ii[-1]}, a_i[-2] = {a_i[-2]}")
                print(f"iteration {i_iter + 1} / {n_iter} with error norm {vnorm(x - x_new):.2g}")
            if i_iter == (n_iter - 1):
                print(f"jacobi did not converge, maximum number of {n_iter} iterations was reached")
        x = x_new
    x_numpy = x.get()
    return x_numpy

def SCSM_jacobi_iter_cupy_old(tri_centers, areas, n, b_im, sig=0.33, omega=1, n_iter=1000, tol=1e-15, verbose=False):

    rs = cp.asarray(tri_centers, dtype='float32')
    areas = cp.asarray(areas, dtype='float32')
    n = cp.asarray(n, dtype='float32')
    b_im = cp.asarray(b_im, dtype='float32')
    M = rs.shape[0]
    eps0 = 8.854187812813e-12

    def vnorm(x):
        return cp.linalg.norm(x)

    def v_vnorm(x):
        x = x.T
        return cp.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2).T

    x = cp.zeros(M, dtype='complex64')
    a_ii = - ((1 / 2) + ((1j * omega * eps0) / sig)) / eps0 / areas
    f = 4 * cp.pi * eps0
    for i_iter in numba.prange(n_iter):
        x_new = cp.zeros_like(x)
        for i in numba.prange(M):
            delta_r = rs[i] - rs
            A11 = (delta_r[:, 0] * n[i, 0]) + (delta_r[:, 1] * n[i, 1]) + (delta_r[:, 2] * n[i, 1])
            A12 = f * v_vnorm(delta_r) ** 3
            a_i = A11 / A12 + 1e-25j
            b_i = 1j * b_im[i]
            s1 = cp.dot(a_i[:i], x[:i])
            s2 = cp.dot(a_i[i + 1:], x[i + 1:])
            x_new[i] = (b_i - s1 - s2) / a_ii[i]
        if vnorm(x - x_new) < tol:
            print(f"jacobi converged after {i_iter + 1} iterations")
            break
        x = x_new
    x_numpy = x.get()
    return x_numpy

def jac_t(tri_centers, areas, n, b_im, M, x, x_new, sig, omega, n_iter, tol):
    f = partial(SCSM_jacobi_iter_cupy_test, xp=cp)
    return f(tri_centers, areas, n, b_im, M, x, x_new, sig, omega, n_iter, tol)

def SCSM_jacobi_iter_cupy_test(man, tri_centers, areas, n, b_im, M, sig, omega, n_iter, tol):

    x = man.np_zeros([M], dtype=np.complex_)
    x_new = man.np_zeros([M], dtype=np.complex_)
    rs = tri_centers
    eps0 = 8.854187812813e-12
    a_ii = - ((1 / 2) + ((1j * omega * eps0) / sig)) / eps0 / areas
    f = 4 * cp.pi * eps0
    multiprocessing.set_start_method('spawn', force=True)
    mp_loop = partial(jac_inner_loop, rs=rs, n=n, b_im=b_im, a_ii=a_ii, f=f, x=x, x_new=x_new)
    partial(jac_inner_loop, xp=cp)
    idx_list_chunks = compute_chunks(list(np.arange(M)), 8)
    for it_count in range(n_iter):
        pool = multiprocessing.Pool(processes=8)
        pool.map(mp_loop, idx_list_chunks)
        pool.close()
        if np.linalg.norm(x - x_new) < tol:
            break
        x = x_new
    return x

def jac_inner_loop(rs, n, b_im, a_ii, f, x, x_new, idxs):
    for k in range(len(idxs)):
        i = idxs[k]
        delta_r = rs[i] - rs
        A11 = (delta_r[:, 0] * n[i, 0]) + (delta_r[:, 1] * n[i, 1]) + (delta_r[:, 2] * n[i, 1])
        v_vnorm_delta_r = cp.sqrt(delta_r.T[0] ** 2 + delta_r.T[1] ** 2 + delta_r.T[2] ** 2).T
        A12 = f * v_vnorm_delta_r ** 3
        a_i = A11 / A12 + 1e-25j
        b_i = 1j * b_im[i]
        s1 = cp.dot(a_i[:i], x[:i])
        s2 = cp.dot(a_i[i + 1:], x[i + 1:])
        x_new[i] = (b_i - s1 - s2) / a_ii[i]


def jacobi_vectors_cupy(rs, n, m=np.array([0, 1, 0]), omega=3e3, m_pos=0):
    rs = cp.asarray(rs, dtype='float32')
    n = cp.asarray(n, dtype='float32')
    m = cp.asarray(m, dtype='float32')
    m_pos = cp.asarray(m_pos, dtype='float32')

    b_im = cp.zeros(rs.shape[0])
    for i in range(m.shape[0]):
        r_r0_norms = cp.linalg.norm(rs - m_pos[i], axis=1)
        b_i = omega * 1e-7 * cp.divide(np.sum(cp.cross(m[i], (rs - m_pos[i])) * n, axis=1), (cp.power(r_r0_norms, 3)))
        b_im = b_im + b_i
    return b_im.get()


def jacobi_vectors_numpy(rs, n, r0=np.array([0, 0, 1.1]), m=np.array([0, 1, 0]), omega=3e3):
    r_r0_norms = np.linalg.norm(rs - r0, axis=1)
    b_im = omega * 1e-7 * np.divide(np.sum(np.cross(m, (rs - r0)) * n, axis=1), (np.power(r_r0_norms, 3)))
    return b_im

@numba.jit(nopython=True, parallel=True)
def jacobi_vectors_numba(b_im, rs, n, r0=np.array([0, 0, 1.1]), m=np.array([0, 1, 0]), omega=3e3, m_pos=0):
    r_r0_norms = np.zeros(rs.shape[0])
    for i in numba.prange(m.shape[0]):
        for j in numba.prange(rs.shape[0]):
            r_r0_norms[j] = np.linalg.norm(rs[j] - m_pos[i])
        b_i = omega * 1e-7 * np.divide(np.sum(np.cross(m[i], (rs - m_pos[i])) * n, axis=1), (np.power(r_r0_norms, 3)))
        b_im = b_im + b_i
        b_out = b_im.copy()
    # print(b_out[0])
    return b_out

def vector_potential_for_E(rs, m=np.array([0, 1, 0]), omega=1, m_pos=0):
    rs = cp.asarray(rs, dtype='float32')
    m = cp.asarray(m, dtype='float32')
    m_pos = cp.asarray(m_pos, dtype='float32')
    b_im = cp.zeros((rs.shape[0], 3))
    for i in range(m.shape[0]):
        r_r0_norms = cp.linalg.norm(rs - m_pos[i], axis=1)
        cross = np.cross(m[i], (rs - m_pos[i]))
        r3 = (cp.power(r_r0_norms, 3))
        b_i = omega * 1e-7 * pfact(1/r3, cross)
        b_im = b_im + b_i
    return b_im.get()

def vector_potential_for_E_single_m(rs, m=np.array([0, 1, 0]), omega=1, m_pos=0):
    rs = cp.asarray(rs, dtype='float32')
    m = cp.asarray(m, dtype='float32')
    m_pos = cp.asarray(m_pos, dtype='float32')
    r_r0_norms = cp.linalg.norm(rs - m_pos, axis=1)
    cross = np.cross(m, (rs - m_pos))
    r3 = (cp.power(r_r0_norms, 3))
    b_i = omega * 1e-7 * np.vstack((cross[:, 0] / r3, cross[:, 1] / r3, cross[:, 2] / r3)).T
    #b_i = omega * 1e-7 * cp.divide(np.cross(m, (rs - m_pos)), (cp.power(r_r0_norms, 3)))
    return b_i.get()


def SCSM_jacobi_iter_debug(tri_centers, areas, n, r0=np.array([0, 0, 1.1]), m=np.array([0, 1, 0]), sig=0.33,
                          omega=1, n_iter=1000, tol=1e-15, A=None, b=None, initial_guess=False, b_im=None,
                           verbose=False):
    rs = tri_centers
    M = rs.shape[0]
    eps0 = 8.854187812813e-12

    def diff(x, y, xlen):
        for i in range(xlen):
            for j in range(3):
                x[i][j] = y[j] - x[i][j]
        return x

    def multiply(x, y, xlen):
        for i in range(xlen):
            x[i] = y * x[i]
        return x

    def dot(x, y, xlen):
        for i in range(xlen):
            x[i] = y[i] * x[i]
        return x

    def div(x):
        for i in range(x.shape[0]):
            x[i] = 1/x[i]
        return x

    def vnorm(x):
        return np.linalg.norm(x)

    def kroen(i, j):
        return int(i == j)

    def a_fun(i, M, n):
        A_i = np.zeros(M, np.complex_)
        for j in numba.prange(M):
            A11 = np.dot((rs[i, :] - rs[j, :]), n[i])
            A12 = (4 * np.pi * eps0 * vnorm(rs[i, :] - rs[j, :]) ** 3 + kroen(i, j))
            A_i[j] = A11 / A12
        return A_i

    x = np.zeros(M, np.complex_)

    if not type(b_im) == np.ndarray:
        b_im = np.zeros(M)
        for i in numba.prange(M):
            b_im[i] = omega * 1e-7 * np.dot(np.cross(m, (rs[i] - r0)), n[i]) / (vnorm(rs[i] - r0) ** 3)

    a_ii = - ((1 / 2) + ((1j * omega * eps0) / sig)) / eps0 / areas
    if initial_guess:
        x = 1j * b_im / a_ii
    for it_count in numba.prange(n_iter):
        # if it_count != 0 and verbose:
        #     print("Iteration {0}: {1}".format(it_count, x))
        x_new = np.zeros_like(x)
        for i in numba.prange(M):
            if not A is None and not b is None:
                a_i = A[i, :]
                b_i = b[i]
            else:
                a_i = a_fun(i, M, n)
                b_i = 1j * b_im[i]

            r = rs
            delta_r = diff(r, r[i], r.shape[0])
            dr0 = delta_r.T[0]
            dr1 = delta_r.T[1]
            dr2 = delta_r.T[2]
            A11 = multiply(dr0, n[i][0], dr0.shape[0]) + multiply(dr1, n[i][1], dr0.shape[0]) + multiply(dr2, n[i][2],
                                                                                                         dr0.shape[0])
            s1 = np.dot(a_i[:i], x[:i])
            s2 = np.dot(a_i[i + 1:], x[i + 1:])
            x_new[i] = (b_i - s1 - s2) / a_ii[i]
            if x_new[i] == x_new[i - 1]:
                break
        if vnorm(x - x_new) < tol:
            break
        x = x_new
    return x

def Q_jacobi_numba_cuda(tc, areas, n_v, b_im, sig=0.33, omega=1, tol=1e-10, n_iter=20):
    x = cuda.to_device(np.zeros(tc.shape[0], dtype=np.complex_))
    out = np.zeros(tc.shape[0], dtype=np.complex_)
    areas = areas + 0j
    a_ic = cuda.to_device(np.zeros(tc.shape[0], dtype=np.complex_))
    x_new = cuda.to_device(np.zeros(tc.shape[0], dtype=np.complex_))
    r_cu = cuda.to_device(tc + 0j)
    a_cu = cuda.to_device(areas + 0j)
    n_cu = cuda.to_device(n_v + 0j)
    b_comp = 1j*b_im
    b_cu = cuda.to_device(b_comp)
    threadsperblock = 32
    blockspergrid = (tc.shape[0] + (threadsperblock - 1)) // threadsperblock
    jac_cu[blockspergrid, threadsperblock](r_cu, a_cu, n_cu, b_cu, a_ic, sig, omega, tol, n_iter, x, x_new, out)
    # cuda.synchronize()
    return out

@cuda.jit()
def jac_cu(r, areas, n, b_im, aic, sig, omega, tol, n_iter, x, x_new, out):

    def diff(x, y, xlen):
        for i in range(xlen):
            for j in range(3):
                x[i][j] = y[j] - x[i][j]
        return x

    def multiply(x, y, xlen):
        for i in range(xlen):
            x[i] = y * x[i]
        return x

    def dot(x, y, xlen):
        sum = 0
        for i in range(xlen):
            #dot-product
            sum += y[i] * x[i]
        return sum

    def div(x):
        for i in range(x.shape[0]):
            x[i] = 1/x[i]
        return x

    def div2(x, y, xlen):
        for i in range(xlen):
            x[i] = x[i] / y[i]
        return x

    def sum(x, y):
        for i in range(x.shape[0]):
            x[i] = x[i] + y[i]
        return x

    def pow(x, y):
        for i in range(x.shape[0]):
            x[i] = x[i] ** y
        return x

    def vdiv(x, y):
        for i in range(x.shape[0]):
            if not y[i] == 0:
                x[i] = x[i] / y[i]
        return x

    def vdiff(x, y, xlen):
        for i in range(xlen):
            x[i] = x[i] - y[i]
        return x

    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    M = r.shape[0]
    eps0 = 8.854187812813e-12
    a_factor = - ((1 / 2) + ((1j * omega * eps0) / sig)) * (1 / eps0)
    a_areas = div(areas)
    a_ii = multiply(a_areas, a_factor, a_areas.shape[0])
    # a_ii = - ((1 / 2) + ((1j * omega * eps0) / sig)) / eps0 * div(areas)
    f = 4 * np.pi * eps0
    xlen = M
    for _ in range(n_iter):
        for i in range(start, M, stride):
            delta_r = diff(r, r[i], r.shape[0])
            dr0 = delta_r.T[0]
            dr1 = delta_r.T[1]
            dr2 = delta_r.T[2]
            drn0 = multiply(dr0, n[i][0], dr0.shape[0])
            drn1 = multiply(dr1, n[i][1], dr1.shape[0])
            drn2 = multiply(dr2, n[i][2], dr2.shape[0])
            A11 = sum(sum(drn0, drn1), drn2)
            A12 = pow((pow(sum(sum(pow(delta_r[0], 2), pow(delta_r[1], 2)), pow(delta_r[2], 2)), 0.5)).T, 3)
            A_12 = multiply(A12, f, dr0.shape[0])
            a_i = vdiv(A11, A_12)
            b_i = b_im[i]
            s1 = dot(a_i[:i], x[:i], a_i[:i].shape[0])
            s2 = dot(a_i[i + 1:], x[i + 1:], a_i[i + 1:].shape[0])
            s3 = (b_i - s1 - s2)
            s4 = 1 / a_ii[i]
            x_new[i] = s3 / s4
        x_diff = vdiff(x, x_new, xlen)
        x_norm = (x_diff[0] ** 2 + x_diff[1] ** 2 + x_diff[2] ** 2) ** 0.5
        if x_norm.imag < tol:
            break
        x = x_new
    out = x


# @cp.fuse
# def jac_cu2(r, areas, n, b_im, sig, omega, M, tol, n_iter, x, x_new):
#
#     def diff(x, y, xlen):
#         for i in range(xlen):
#             for j in range(3):
#                 x[i][j] = y[j] - x[i][j]
#         return x
#
#     def multiply(x, y, xlen):
#         for i in range(xlen):
#             x[i] = y * x[i]
#         return x
#
#     def dot(x, y, xlen):
#         sum = 0
#         for i in range(xlen):
#             #dot-product
#             sum += y[i] * x[i]
#         return sum
#
#     def div(x):
#         for i in range(x.shape[0]):
#             x[i] = 1/x[i]
#         return x
#
#     def div2(x, y, xlen):
#         for i in range(xlen):
#             x[i] = x[i] / y[i]
#         return x
#
#     def sum(x, y):
#         for i in range(xlen):
#             x[i] = x[i] + y[i]
#         return x
#
#     def pow(x, y, xlen):
#         for i in range(xlen):
#             x[i] = x[i] ** y
#         return x
#
#     def vdiv(x, y, xlen):
#         for i in range(xlen):
#             if not y[i] == 0:
#                 x[i] = x[i] / y[i]
#         return x
#
#     def vdiff(x, y, xlen):
#         for i in range(xlen):
#             x[i] = x[i] - y[i]
#         return x
#
#     eps0 = 8.854187812813e-12
#     xlen = M
#     a_factor = - ((1 / 2) + ((1j * omega * eps0) / sig)) * (1 / eps0)
#     a_areas = div(areas)
#     a_ii = multiply(a_areas, a_factor, xlen)
#     f = 4 * np.pi * eps0
#     for _ in range(n_iter):
#         for i in range(M):
#             delta_r = diff(r, r[i], xlen)
#             dr0 = delta_r.T[0]
#             dr1 = delta_r.T[1]
#             dr2 = delta_r.T[2]
#             drn0 = multiply(dr0, n[i][0], xlen)
#             drn1 = multiply(dr1, n[i][1], xlen)
#             drn2 = multiply(dr2, n[i][2], xlen)
#             A11 = sum(sum(drn0, drn1), drn2)
#             A12 = pow((pow(sum(sum(pow(delta_r[0], 2), pow(delta_r[1], 2)), pow(delta_r[2], 2)), 0.5)).T, 3)
#             A_12 = multiply(A12, f, xlen)
#             a_i = vdiv(A11, A_12)
#             b_i = b_im[i]
#             s1 = dot(a_i[:i], x[:i], xlen)
#             s2 = dot(a_i[i + 1:], x[i + 1:], xlen)
#             s3 = (b_i - s1 - s2)
#             s4 = 1 / a_ii[i]
#             x_new[i] = s3 / s4
#         x_diff = vdiff(x, x_new, xlen)
#         x_norm = (x_diff[0] ** 2 + x_diff[1] ** 2 + x_diff[2] ** 2) ** 0.5
#         if x_norm.imag < tol:
#             break
#         x = x_new
#     return x

# def SCSM_tri_sphere_dask(tri_centers, tri_points, areas, r0=np.array([0, 0, 1.1]), m=np.array([0, 1, 0]), sig=0.33, omega=1):
#     rs = tri_centers
#     M = rs.shape[0]
#     A = da.zeros((M, M), dtype=complex)
#     B = da.zeros(M, dtype=complex)
#     eps0 = 8.854187812813e-12
#
#     def vnorm(x):
#         return np.linalg.norm(x)
#
#     def kroen(i, j):
#         return int(i == j)
#
#     for i in range(M):
#         r_norm_i = rs[i] / vnorm(rs[i])
#         p1 = tri_points[i][0]
#         p2 = tri_points[i][1]
#         p3 = tri_points[i][2]
#         n = - np.cross((p3 - p1), (p2 - p3)) / (2 * areas[i])
#         for j in range(M):
#             A11 = np.dot((rs[i, :] - rs[j, :]), n)
#             A12 = (4 * np.pi * eps0 * vnorm(rs[i, :] - rs[j, :]) ** 3 + kroen(i, j))
#             A1 = A11 / A12
#             A2 = kroen(i, j) / (eps0 * areas[i]) * ((1 / 2) + ((1j * omega * eps0) / sig))
#             A[i, j] = A1 - A2
#         B[i] = 1j * omega * 1e-7 * np.dot(np.cross(m, (rs[i] - r0)), n) / (vnorm(rs[i] - r0) ** 3)
#     # B = 1j * omega * 1e-7 * np.dot(np.cross(m, (rs - r0)), (rs / vnorm(rs))) / (vnorm(rs - r0) ** 3)
#     Q = da.linalg.solve(A, B)
#     return Q.compute(), rs


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

def SCSM_FMM_E(Q, r_source, r_target, eps, m=np.array([0, 1, 0]), r0=np.array([0, 0, 1.1]), omega=1,
                        projection="polar"):
    eps0 = 8.854187812813e-12
    n = r_target.shape[0]
    E = np.zeros(n)
    charges = Q.imag
    fmm_res_phi = fmm.lfmm3d(eps=eps, sources=r_source.T, charges=charges, pg=2, pgt=2, targets=r_target.T)
    grad_phi = -1/(4 * np.pi * eps0) * fmm_res_phi.gradtarg.T
    for i in range(n):
                r_v = r_target[i]
                E_v = grad_phi[i] - (omega * 1e-7) * (np.cross(m, (r_v - r0))) / (vnorm(r_v - r0) ** 3)
                E[i] = vnorm(E_v)
    return E

def SCSM_FMM_E2(Q, r_source, r_target, eps, b_im):
    eps0 = 8.854187812813e-12
    n = r_target.shape[0]
    E = np.zeros(n)
    charges = Q.imag
    fmm_res_phi = fmm.lfmm3d(eps=eps, sources=r_source.T, charges=charges, pg=2, pgt=2, targets=r_target.T)
    grad_phi = -1/(4 * np.pi * eps0) * fmm_res_phi.gradtarg.T
    E_v = grad_phi - b_im
    E = np.linalg.norm(E_v, axis=1)
    return E

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
                        phi=np.zeros((10, 10))):
    eps0 = 8.854187812813e-12
    I = phi.shape[0]
    J = theta.shape[0]
    N = r_q.shape[0]
    E = np.zeros((phi.shape[0], theta.shape[0]))

    def vnorm(x):
        return np.linalg.norm(x)

    for i in numba.prange(I):
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
            E_complex = grad_phi.sum(axis=1) - (1j * omega * 1e-7) * (np.cross(m, (r_v - r0))) / (
                    vnorm(r_v - r0) ** 3)
            E[i, j] = vnorm(E_complex.imag)
    return E

@numba.jit(nopython=True, parallel=True)
def E_near_correction(E, Q, r_q, r_sphere, tri_points, theta, phi=np.zeros((10, 10)), n=0, r_near=3e-1):
    eps0 = 8.854187812813e-12
    I = phi.shape[0]
    J = theta.shape[0]
    E_new = np.zeros((phi.shape[0], theta.shape[0]))

    def vnorm(x):
        return np.linalg.norm(x)

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
            f12x = np.dot(n, k12x) / s12 * np.log((d1 + d2 + s12) / (d1 + d2 - s12))
        if s23 == 0 or s23 == d2 + d3:
            f23x = 0
        else:
            f23x = np.dot(n, k23x) / s23 * np.log((d2 + d3 + s23) / (d2 + d3 - s23))
        if s31 == 0 or s31 == d3 + d1:
            f31x = 0
        else:
            f31x = np.dot(n, k31x) / s31 * np.log((d3 + d1 + s31) / (d3 + d1 - s31))
        # f for y component
        if s12 == 0 or s12 == d1 + d2:
            f12y = 0
        else:
            f12y = np.dot(n, k12y) / s12 * np.log((d1 + d2 + s12) / (d1 + d2 - s12))
        if s23 == 0 or s23 == d2 + d3:
            f23y = 0
        else:
            f23y = np.dot(n, k23y) / s23 * np.log((d2 + d3 + s23) / (d2 + d3 - s23))
        if s31 == 0 or s31 == d3 + d1:
            f31y = 0
        else:
            f31y = np.dot(n, k31y) / s31 * np.log((d3 + d1 + s31) / (d3 + d1 - s31))
        # f for z component
        if s12 == 0 or s12 == d1 + d2:
            f12z = 0
        else:
            f12z = np.dot(n, k12z) / s12 * np.log((d1 + d2 + s12) / (d1 + d2 - s12))
        if s23 == 0 or s23 == d2 + d3:
            f23z = 0
        else:
            f23z = np.dot(n, k23z) / s23 * np.log((d2 + d3 + s23) / (d2 + d3 - s23))
        if s31 == 0 or s31 == d3 + d1:
            f31z = 0
        else:
            f31z = np.dot(n, k31z) / s31 * np.log((d3 + d1 + s31) / (d3 + d1 - s31))
        # perpendicular components
        arc_tangent_sum = np.arctan2(D1, N * d1) + np.arctan2(D2, N * d2) + np.arctan2(D3, N * d3) + np.sign(h) * np.pi
        arc_tangent_sum = np.arctan(D1 / (N * d1)) + np.arctan(D2 / (N * d2)) + np.arctan(D3 / (N * d3)) + np.sign(
            h) * np.pi / 2
        gx = - n[0] * arc_tangent_sum
        gy = - n[1] * arc_tangent_sum
        gz = - n[2] * arc_tangent_sum

        Ex = f12x + f23x + f31x + gx
        Ey = f12y + f23y + f31y + gy
        Ez = f12z + f23z + f31z + gz

        E = -Q / (A * 4 * np.pi * eps0) * np.array((Ex, Ey, Ez))
        return E

    for i in numba.prange(I):
        for j in numba.prange(J):
            phi_i = phi.T[0][i]
            theta_j = theta[0][j]
            x = r_sphere * np.sin(phi_i) * np.cos(theta_j)
            y = r_sphere * np.sin(phi_i) * np.sin(theta_j)
            z = r_sphere * np.cos(phi_i)
            r_v = np.array((x, y, z))
            E_1 = np.zeros(n)
            E_2 = np.zeros(n)
            for n in numba.prange(n):
                if vnorm(r_v - r_q[n]) < r_near:
                    E_1[n] = vnorm(Q[n].imag * (r_v - r_q[n]) / (4 * np.pi * eps0 * vnorm(r_v - r_q[n]) ** 3))
                    E_2[n] = vnorm(E_near(Q[n], tri_points[n][0], tri_points[n][1], tri_points[n][2], r_v).imag)
            E_prev = E[i, j]
            E_new[i, j] = E_prev - E_1.sum() + E_2.sum()
    return E_new

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
    for i in range(I):
        for j in range(J):

            if projection == "polar":
                xs, ys = r_sphere * np.cos(theta[j]), r_sphere * np.sin(theta[j])
                rs = np.array([xs, ys, np.zeros(xs.shape[0])])
                r_v = rs[:, i]
            elif projection == "sphere_surface":
                phi_i = phi[i, 0]
                theta_j = theta[0, j]
                x, y, z = r_sphere * np.sin(phi_i) * np.cos(theta_j), r_sphere * np.sin(phi_i) * np.sin(theta_j),\
                          r_sphere * np.cos(phi_i)
                r_v = np.array([x, y, z])
            else:
                raise TypeError("projection can only be 'polar' or 'sphere_surface'!")

            grad_phi = np.zeros([3, N], dtype=np.complex_)
            for n in range(N):
                if near_field:
                    if near_field:
                        # eps_r = vnorm(r_q[n] - r_v)
                        # if eps_r < near_radius:
                        grad_phi[:, n] = E_near(Q[n], tri_points[n][0], tri_points[n][1], tri_points[n][2], r_v)
                else:
                    grad_phi[:, n] = Q[n] * (r_v - r_q[n]) / (4 * np.pi * eps0 * vnorm(r_v - r_q[n]) ** 3)
            E_complex = grad_phi.sum(axis=1) - (1j * omega * 1e-7) * (np.cross(m, (r_v - r0))) / (vnorm(r_v - r0) ** 3)
            E[i, j] = vnorm(E_complex.imag)

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

def get_n(tri_points, areas):
    p1 = np.vstack((tri_points.T[0][0], tri_points.T[1][0], tri_points.T[2][0])).T
    p2 = np.vstack((tri_points.T[0][1], tri_points.T[1][1], tri_points.T[2][1])).T
    p3 = np.vstack((tri_points.T[0][2], tri_points.T[1][2], tri_points.T[2][2])).T
    n_1 = np.cross((p3 - p1), (p2 - p3))
    n = - np.vstack((n_1.T[0] / (2 * areas), n_1.T[1] / (2 * areas), n_1.T[2] / (2 * areas))).T
    n = np.ascontiguousarray(n)
    return n

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
        # y goes from r to -r
        y = 1 - (i / float(samples - 1)) * 2
        # radius at y
        radius = math.sqrt(1 - y * y)
        # golden angle increment
        theta = phi * i

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points)

def array_unflatten(array, n_rows=1):

    n_cols = int(array.shape[0] / n_rows)
    new_array = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        new_array[i, :] = array[(i * n_cols):((i + 1) * n_cols)]
    return new_array

def array_unflatten3d(array, n_rows=1):

    n_cols = int(array.shape[0] / n_rows)
    new_array = np.zeros((n_rows, n_cols, 3))
    for i in range(n_rows):
        new_array[i, :, :] = array[(i * n_cols):((i + 1) * n_cols)]
    return new_array

def t_format(time):
    if time < 60:
        return time, "s"
    else:
        t_min = time / 60
        if t_min < 60:
            return t_min, "min"
        else:
            t_h = t_min / 60
            if t_h < 24:
                return t_h, "h"
            else:
                t_d = t_h / 24
                return t_d, "d"

def Jacobi_iter(A, b, n_iter, tol, rho=0, verbose=False):

    x = np.zeros_like(b)
    for it_count in range(n_iter):
        if it_count != 0 and rho < 1 and verbose:
            print("Iteration {0}: {1}".format(it_count, x))
        x_new = np.zeros_like(x)

        for i in range(A.shape[0]):
            a_i = A[i, :]
            a_ii = A[i, i]
            s1 = np.dot(a_i[:i], x[:i])
            s2 = np.dot(a_i[i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / a_ii
            if x_new[i] == x_new[i - 1]:
                break

        if np.allclose(x, x_new, atol=tol, rtol=0.):
            break

        x = x_new

    return x

def spectral_radius(A):
    M = A.shape[0]
    U = np.triu(A, k=1)
    L = np.tril(A, k=-1)
    D = np.zeros_like(A)
    D.flat[0::M + 1] = np.diag(A)
    T = np.linalg.inv(D) @ (L + U)
    lambdas = np.linalg.eigvals(T)
    rho = np.linalg.norm(np.max(lambdas))
    return rho

def save_to_hdf5(fn = "file", matrices = [], names = []):
    hf = h5py.File(fn + '.h5', 'w')
    for i in range(len(matrices)):
        hf.create_dataset(names[i], data=matrices[i])
        print(f"{names[i]} saved")


def read_ccd(data_path, fn):
    ccd_file = os.path.join(data_path, fn)
    with open(ccd_file) as f:
        lines = f.readlines()

    f.close()

    lines = [i.replace(" ", ", ") for i in lines]
    lines = [i.replace(", \n", "") for i in lines]
    lines = [i.replace("'", "") for i in lines]
    lines = [i.split(",") for i in lines[2:]]
    for j in range(len(lines)):
        lines[j] = [float(i) for i in lines[j]]

    data = np.asarray(lines)
    r0 = data[:3]
    m = data[3:]
    return r0, m

def read_from_ccd(data_path):

    ccd_file = os.path.join(data_path, 'MagVenture_MCF_B65_REF_highres.ccd')
    with open(ccd_file) as f:
        lines = f.readlines()

    f.close()

    lines = [i.replace(" ", ", ") for i in lines]
    lines = [i.replace(", \n", "") for i in lines]
    lines = [i.replace("'", "") for i in lines]
    lines = [i.split(",") for i in lines[2:]]
    for j in range(len(lines)):
        lines[j] = [float(i) for i in lines[j]]

    data = np.asarray(lines)
    m_pos = data[:, :3]
    m = data[:, :-3]
    return m, 1000 * m_pos

def array_3d_plot(array, array1=0):

    xs = array[:, 0]
    ys = array[:, 1]
    zs = array[:, 2]
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(xs, ys, zs)
    if type(array1) == np.ndarray:
        xs1 = array1[:, 0]
        ys1 = array1[:, 1]
        zs1 = array1[:, 2]
        ax.scatter3D(xs1, ys1, zs1)
    plt.show()

def translate(array, transformation_matrix):
    a = np.vstack((array.T, np.ones(array.shape[0]))).T
    a = a @ transformation_matrix
    res = a[:, :3]
    return res


def layered_sphere_mesh(n_samples, sigmas, radii=np.array([0.8, 0.9, 0.905, 0.91, 1.0])):

    for n_iter in range(radii.shape[0]):
        tc_n, areas_n, tri_points_n, n_v_n, avg_lens_n = sphere_mesh(n_samples, scaling=radii[n_iter])
        if n_iter == 0:
            div = tc_n.shape[0]
            tc = tc_n
            areas = areas_n
            tri_points = tri_points_n
            n_v = n_v_n
            avg_lens = np.array([avg_lens_n])
        if n_iter > 0:
            tc = np.vstack((tc, tc_n))
            areas = np.concatenate((areas, areas_n))
            tri_points = np.vstack((tri_points, tri_points_n))
            n_v = np.vstack((n_v, n_v_n))
            avg_lens = np.concatenate((avg_lens, np.array([avg_lens_n])))
    avg_lens = np.mean(avg_lens)
    # set up realistic sigma values
    sigmas_in = np.zeros(tc.shape[0])
    sigmas_out = np.zeros_like(sigmas_in)
    for i in range(sigmas.shape[0]):

        if i < (sigmas.shape[0] - 1):
            sigmas_in[((i) * div):((i + 1) * div)] = sigmas[i]
            sigmas_out[((i) * div):((i + 1) * div)] = sigmas[i + 1]
        else:
            sigmas_in[((i) * div):((i + 1) * div)] = sigmas[i]
            sigmas_out[((i) * div):((i + 1) * div)] = 0.0

    return tc, areas, tri_points, n_v, avg_lens, sigmas_in, sigmas_out


def xyz_grid(r, phi, theta):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z])

def nrmse(refference, res):
    diff = np.subtract(res, refference)
    square = np.square(diff)
    mse = square.mean()
    rmse = np.sqrt(mse)
    nrmse = rmse/(np.max(res)-np.min(res))
    return nrmse

def v_vnorm(x):
    x = x.T
    return np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2).T

def radial_field_norm(E):
    E_r = v_vnorm(E)
    return E_r

def pdot(a, b):
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]

def pfact(f, array):
    return np.vstack((f[:] * array[:, 0], f[:] * array[:, 1],  f[:] * array[:, 2])).T

def Norm_x_y(array, n):
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = np.linalg.norm(array[i, j, :])
    return out


def One_layer_sphere_single_m_test(n, r_out, r_in, m, direction, omega, n_samples, scaling_factor, method, tol=1e16, n_iter=20,
                                   print_time=True):
    start = time.time()

    # set up ROI grid
    phi1 = np.linspace(0, np.pi, n)
    theta1 = np.linspace(0, 2 * np.pi, n)
    phi2, theta2 = np.meshgrid(phi1, theta1)
    phi, theta = phi2.T, theta2.T
    r = r_in * scaling_factor
    grid = xyz_grid(r, phi, theta)

    # generate dipole location
    d_norm = direction / np.linalg.norm(direction)
    r0 = r_out * d_norm * scaling_factor

    # convert mesh of ROI to target location vectors
    r_target = sphere_to_carthesian(r=r, phi=phi.flatten(), theta=theta.flatten())

    # calculate analytic solution
    res1_vector_flat, res1_flat = reciprocity_surface_single_m(rs=r_target, r0_v=r0, m=m, omega=omega)
    res1 = array_unflatten(res1_flat, n_rows=n)
    # res1 = reciprocity_sphere(grid=xyz_grid, r0_v=r0, m=m, omega=omega)[1]

    # tc, areas = functions.read_sphere_mesh_from_txt(sizes, path)
    # tc, areas, tri_points = read_sphere_mesh_from_txt_locations_only(sizes, path, scaling=scaling_factor)
    tc, areas, tri_points, n_v, avg_len = sphere_mesh(n_samples, scaling=scaling_factor)
    print(f"average length: {avg_len}")
    n_elements = tc.shape[0]
    print(f"elements: {n_elements}")

    # Q, rs = SCSM_tri_sphere_numba(tc, tri_points, areas, r0=r0, m=m, omega=omega)
    b_im = jacobi_vectors_numpy(tc, n_v, r0, m, omega=omega)
    if method == "jacobi":
        Q = SCSM_jacobi_iter_cupy(tc, areas, n_v, b_im, tol=tol, n_iter=n_iter, omega=omega)
    elif method == "matrix":
        Q = SCSM_matrix(tc, areas, n=n_v, b_im=b_im, omega=omega)
    # Q = SCSM_tri_sphere_numba(tc, tri_points, areas, r0=r0, m=m, omega=3e3)[0]
    b_im_ = vector_potential_for_E_single_m(rs=r_target, m=m, m_pos=r0, omega=omega)
    res_flat = SCSM_FMM_E2(Q=Q, r_source=tc, r_target=r_target, eps=1e-15, b_im=b_im_)
    res = array_unflatten(res_flat, n_rows=n)
    end = time.time()
    t = t_format(end - start)
    if print_time:
        print(f"simulation time needed: {t[0]:.2f}" + t[1])

    return res1, res, grid


def plot_curve(x, y):
    plt.plot(x, y)
    plt.show()