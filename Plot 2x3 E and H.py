import functions
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

ax = plt.subplots(2, 3, projection='polar')

loc_list_1 = [0, 0, 0, 1, 1, 1]
loc_list_2 = [0, 1, 2, 0, 1, 2]
r_list = [np.array([12, 0, 0]), np.array([4, 11, 0]), np.array([0, 12, 0]), np.array([-4, 11, 0]),
          np.array([-12, 0, 0]), np.array([0, -12, 0])]
for i in range(2):
    for j in range(3):

        r = np.linspace(0.01, 8, 400)
        theta = np.linspace(0, 2*np.pi, 400)
        line1 = 7*np.ones(400)
        line2 = 7.5*np.ones(400)
        line3 = 8*np.ones(400)

        res = functions.reciprocity_three_D(r, theta, r0_v=r_list[i+j], m=np.array([0, 1, 0]))
        # res_mag = func_3_shells(r, theta, r0_v=np.array([4, 11, 0]))
        # res = func_de_Munck_potential(r, theta)
        f_min, f_max = res.min(), res.max()
        # f_min, f_max = res_mag.min(), res_mag.max()
        ax_ij = ax[i, j]
        im = ax_ij.pcolormesh(theta, r, res, cmap='plasma', vmin=f_min, vmax=f_max)
        # im = ax.pcolormesh(theta, r, res_mag, cmap='plasma', vmin=f_min, vmax=f_max)
        ax_ij.set_yticklabels([])
        ax_ij.set_rmax(8)
        ax_ij.plot(theta, line1, c='k')
        ax_ij.plot(theta, line2, c='k')
        ax_ij.plot(theta, line3, c='k')
        ax_ij.grid(True)