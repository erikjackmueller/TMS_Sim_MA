import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

def fun(phi, theta, n):
    res = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x, y, z = np.sin(phi[i])*np.cos(theta[j]), np.cos(phi[i])*np.cos(theta[j]), np.cos(theta[j])
            res[i, j] = x**2 - y**2 + z
    return res


r = 0.05
N = 100
phi, theta = np.linspace(0, 2*np.pi, N), np.linspace(0, 2*np.pi, N)
# u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:100j]
u, v = np.meshgrid(phi, theta)
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
res = fun(phi, theta, N)
fcolors = res
fmax, fmin = fcolors.max(), fcolors.min()
fcolors = (fcolors - fmin)/(fmax - fmin)
# ax.plot_surface(x, y, z, facecolors=facecolors, cmap=cm.coolwarm)
fig = plt.figure(figsize=plt.figaspect(1.))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=cm.coolwarm(fcolors))
# ax.grid()
m = cm.ScalarMappable(cmap=cm.coolwarm)
fig.colorbar(m, shrink=0.8)
plt.show()