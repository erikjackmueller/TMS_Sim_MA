import matplotlib.pyplot as plt
import numpy as np

def fun(phi, theta):
    res = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            x, y, z = np.sin(phi[i])*np.cos(theta[j]), np.cos(phi[i])*np.cos(theta[j]), np.cos(theta[j])
            res[i, j] = x**2 - y**2 + z
    return res

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
r = 0.05
u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:100j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
# res = fun(u, v)
facecolors = plt.cm.jet(np.ones((100, 100)))
ax.plot_surface(x, y, z, facecolors=facecolors)
plt.show()